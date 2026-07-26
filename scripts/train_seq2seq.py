"""Unified ENCODER-FREE SEQ2SEQ trainer for the controlled MeanFlow-vs-diffusion study.

One script, one shared recipe, one shared backbone. Only --objective changes:

    --objective meanflow    UNetGenerator, velocity loss, 1-NFE block prediction
    --objective diffusion   UNetGenerator (same config), predict-x0 Huber, DDIM block
    --objective regression  JointRegressor (ConvLSTM seq2seq) -- discriminative baseline

--size {small,large} picks the matched backbone capacity (small = no attention).
Shared: global-std normalization, Adam + cosine-warmup LR, EMA, grad clip, the same
data stream, and per-step NMSE eval at 20 dB. So a MeanFlow-vs-diffusion difference
is attributable ONLY to the objective.

Usage (Alvis):
    python -m scripts.train_seq2seq --objective meanflow  --size large --out-dir runs/s2s_mf_L
    python -m scripts.train_seq2seq --objective diffusion --size large --out-dir runs/s2s_di_L
    python -m scripts.train_seq2seq --objective regression --out-dir runs/s2s_reg
"""

from __future__ import annotations

import argparse, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset, make_fixed_eval_set
from mf_csi.data.dataset import estimate_global_std
from mf_csi.models import UNetGenerator, JointRegressor
from mf_csi.diffusion import make_scheduler, corrupt_history
from mf_csi.seq2seq import (seq2seq_unet_config,
                            meanflow_seq2seq_loss, meanflow_seq2seq_predict,
                            diffusion_seq2seq_loss, diffusion_seq2seq_predict)
from mf_csi.inference import nmse, nmse_db
from mf_csi.ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["meanflow", "diffusion", "regression"])
    p.add_argument("--size", default="large", choices=["small", "large"])
    p.add_argument("--out-dir", required=True)
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


def build_net(objective, size, cfg):
    if objective == "regression":
        return JointRegressor(cfg.regression)
    ucfg = seq2seq_unet_config(cfg.data.num_past, cfg.data.num_future,
                               ch_mult=(2 if size == "large" else 1),
                               num_res_blocks=(2 if size == "large" else 1),
                               use_attention=(size == "large"))
    return UNetGenerator(ucfg)


@torch.no_grad()
def evaluate(net, val_batches, cfg, device, objective, scheduler):
    snr = cfg.train.eval_snr_db
    ps_sum = None
    for b in val_batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        Nf = future.shape[1]
        if objective == "meanflow":
            pred = meanflow_seq2seq_predict(net, hist, Nf, seed_std=cfg.inference.seed_std)
        elif objective == "diffusion":
            pred = diffusion_seq2seq_predict(net, scheduler, hist, Nf, cfg.diu)
        else:
            pred = net(hist)
        ps, _ = nmse(pred, future)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    ps = ps_sum / len(val_batches)
    return ps, ps.mean()


def train_loss(net, past, future, cfg, objective, scheduler, huber):
    if objective == "meanflow":
        return meanflow_seq2seq_loss(net, past, future, cfg.meanflow)
    if objective == "diffusion":
        return diffusion_seq2seq_loss(net, scheduler, past, future, cfg.diu, huber)
    # regression: MSE on the block, with scale-preserving history noise
    hist = corrupt_history(past, cfg.regression.snr_db_min, cfg.regression.snr_db_max)
    pred = net(hist)
    loss = F.mse_loss(pred, future)
    return loss, {"loss": loss.detach()}


def save_ckpt(path, step, net, ema, opt, best, global_ab, meta):
    torch.save({"step": step, "best_nmse_db": best, "global_ab": global_ab, "meta": meta,
                "net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict()}, path)


def main():
    args = parse_args()
    cfg = Config()
    cfg.data.normalization = "global_std"
    cfg.train.total_steps = args.steps
    cfg.train.batch_size = args.batch_size
    cfg.train.eval_every = args.eval_every
    cfg.train.ckpt_every = args.ckpt_every
    cfg.inference.seed_std = cfg.meanflow.source_std          # sigma=1 (global-std -> unit var)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = build_net(args.objective, args.size, cfg).to(device)
    net_eval = build_net(args.objective, args.size, cfg).to(device)
    scheduler = make_scheduler(cfg.diu)
    huber = nn.HuberLoss(delta=cfg.diu.huber_delta)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"objective={args.objective} size={args.size} params={n_params/1e6:.3f}M "
          f"steps={args.steps} batch={args.batch_size} lr={args.lr} norm=global_std", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           betas=(cfg.train.adam_beta1, cfg.train.adam_beta2))
    ema = EMA(net, cfg.train.ema_decay)

    start_step, best = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1; best = ck.get("best_nmse_db", float("inf"))
        print(f"resumed from {args.resume} at step {start_step}")

    ck0 = ck.get("global_ab") if (args.resume and os.path.isfile(args.resume)) else None
    global_ab = ck0 if ck0 is not None else estimate_global_std(cfg.data, num_samples=4000, batch_size=256)
    print(f"global-std (mu, sd) = ({global_ab[0]:.4f}, {global_ab[1]:.4f})")

    val_batches = make_fixed_eval_set(cfg.data, cfg.train.val_samples,
                                      cfg.train.val_batch_size, global_ab=global_ab)
    train_iter = iter(DataLoader(
        CSIStreamDataset(cfg.data, batch_size=cfg.train.batch_size, steps_per_epoch=None,
                         global_ab=global_ab), batch_size=None))
    meta = {"objective": args.objective, "size": args.size}

    net.train()
    t0 = time.time(); running, running_n = 0.0, 0
    for step in range(start_step, cfg.train.total_steps):
        batch = next(train_iter)
        past, future = batch["past"].to(device), batch["future"].to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, cfg.train.warmup_steps, cfg.train.total_steps)
        loss, m = train_loss(net, past, future, cfg, args.objective, scheduler, huber)
        opt.zero_grad(); loss.backward()
        if cfg.train.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip_norm)
        opt.step()
        eff = min(cfg.train.ema_decay, (1.0 + step) / (10.0 + step))
        ema.update(net, eff)
        running += m["loss"].item(); running_n += 1

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:6d} | loss {running/running_n:.4f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {rate:.1f} it/s", flush=True)
            running, running_n = 0.0, 0

        if step > 0 and step % cfg.train.eval_every == 0:
            ema.copy_to(net_eval)
            per_step, overall = evaluate(net_eval, val_batches, cfg, device, args.objective, scheduler)
            avg_db = nmse_db(overall).item()
            s1, sN = nmse_db(per_step[0]).item(), nmse_db(per_step[-1]).item()
            print(f"  [eval] step {step} | EMA avg NMSE {avg_db:.2f} dB "
                  f"(step1 {s1:.2f}, step{len(per_step)} {sN:.2f})", flush=True)
            if avg_db < best:
                best = avg_db
                save_ckpt(os.path.join(args.out_dir, "ckpt_best.pt"),
                          step, net, ema, opt, best, global_ab, meta)
        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(os.path.join(args.out_dir, "ckpt_last.pt"),
                      step, net, ema, opt, best, global_ab, meta)

    save_ckpt(os.path.join(args.out_dir, "ckpt_last.pt"),
              cfg.train.total_steps - 1, net, ema, opt, best, global_ab, meta)
    print(f"done. best avg NMSE {best:.2f} dB. checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
