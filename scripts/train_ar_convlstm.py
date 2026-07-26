"""Train the standalone autoregressive ConvLSTM baseline (next-frame MSE).

Mirrors the other AR trainers (warmup+cosine LR, grad clip, EMA, periodic AR-NMSE eval,
checkpoint/resume). Trains on a randomly-split next-frame target so the model is robust
to the growing/imperfect history it sees during the AR rollout. std normalization.

Usage (Alvis):
    python -m scripts.train_ar_convlstm --out-dir runs/arlstm --steps 50000
"""

from __future__ import annotations

import argparse, math, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset, make_fixed_eval_set
from mf_csi.models import ARConvLSTM
from mf_csi.diffusion import corrupt_history
from mf_csi.inference import ar_convlstm_predict, nmse, nmse_db
from mf_csi.ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="runs/arlstm")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def lr_at(step, peak, warm, total):
    if warm > 0 and step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


def random_split_batch(past, future):
    full = torch.cat([past, future], dim=1)
    T = full.shape[1]
    t_in = int(torch.randint(1, T, (1,)).item())
    return full[:, :t_in], full[:, t_in]


@torch.no_grad()
def evaluate(model, val_batches, cfg, device):
    snr = cfg.train.eval_snr_db
    ps_sum = None
    for b in val_batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = ar_convlstm_predict(model, hist, future.shape[1])
        ps, _ = nmse(pred, future)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    ps = ps_sum / len(val_batches)
    return ps, ps.mean()


def save_ckpt(path, step, model, ema, opt, best):
    torch.save({"step": step, "best_nmse_db": best, "model": model.state_dict(),
                "ema": ema.state_dict(), "opt": opt.state_dict()}, path)


def main():
    args = parse_args()
    cfg = Config()
    cfg.data.normalization = "std"
    cfg.train.total_steps = args.steps
    cfg.train.batch_size = args.batch_size
    cfg.train.eval_every = args.eval_every
    cfg.train.ckpt_every = args.ckpt_every
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ARConvLSTM(cfg.ar_convlstm).to(device)
    model_eval = ARConvLSTM(cfg.ar_convlstm).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"AR-ConvLSTM params={n/1e6:.3f}M steps={args.steps} batch={args.batch_size} lr={args.lr}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           betas=(cfg.train.adam_beta1, cfg.train.adam_beta2))
    ema = EMA(model, cfg.train.ema_decay)

    start_step, best = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1; best = ck.get("best_nmse_db", float("inf"))
        print(f"resumed at {start_step}")

    val_batches = make_fixed_eval_set(cfg.data, cfg.train.val_samples, cfg.train.val_batch_size)
    train_iter = iter(DataLoader(
        CSIStreamDataset(cfg.data, batch_size=cfg.train.batch_size, steps_per_epoch=None),
        batch_size=None))

    model.train()
    t0 = time.time(); running, running_n = 0.0, 0
    for step in range(start_step, cfg.train.total_steps):
        batch = next(train_iter)
        past, future = batch["past"].to(device), batch["future"].to(device)
        history, target = random_split_batch(past, future)
        hist = corrupt_history(history, cfg.ar_convlstm.snr_db_min, cfg.ar_convlstm.snr_db_max) \
            if cfg.ar_convlstm.noise_aug else history
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, cfg.train.warmup_steps, cfg.train.total_steps)
        pred = model(hist)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward()
        if cfg.train.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        opt.step()
        eff = min(cfg.train.ema_decay, (1.0 + step) / (10.0 + step))
        ema.update(model, eff)
        running += loss.item(); running_n += 1

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:6d} | mse {running/running_n:.4f} | lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {rate:.1f} it/s", flush=True)
            running, running_n = 0.0, 0
        if step > 0 and step % cfg.train.eval_every == 0:
            ema.copy_to(model_eval)
            per_step, overall = evaluate(model_eval, val_batches, cfg, device)
            avg_db = nmse_db(overall).item()
            print(f"  [eval] step {step} | EMA avg NMSE {avg_db:.2f} dB "
                  f"(step1 {nmse_db(per_step[0]).item():.2f}, step{len(per_step)} {nmse_db(per_step[-1]).item():.2f})",
                  flush=True)
            if avg_db < best:
                best = avg_db
                save_ckpt(os.path.join(args.out_dir, "ckpt_best.pt"), step, model, ema, opt, best)
        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(os.path.join(args.out_dir, "ckpt_last.pt"), step, model, ema, opt, best)

    save_ckpt(os.path.join(args.out_dir, "ckpt_last.pt"), cfg.train.total_steps - 1, model, ema, opt, best)
    print(f"done. best avg NMSE {best:.2f} dB. in {args.out_dir}")


if __name__ == "__main__":
    main()
