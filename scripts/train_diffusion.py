"""Train the diffusion DiU (ConvLSTM predictor + diffusers UNet2DModel, DDIM).

Mirrors scripts/train.py (warmup, grad clip, EMA, periodic AR-NMSE eval,
checkpoint/resume) but with the diffusion objective. Training predicts a single
next frame from a random-length history window (as in the paper's code); eval
does the full autoregressive DDIM rollout on the fixed Np-frame history.

Usage:
    python -m scripts.train_diffusion --out-dir runs/diu1 --steps 50000
"""

from __future__ import annotations

import argparse
import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset, make_fixed_eval_set, estimate_global_minmax
from mf_csi.models.diu import DiUEncoder, DiUNet
from mf_csi.diffusion import make_scheduler, diffusion_loss, ddim_ar_predict, corrupt_history
from mf_csi.inference import nmse, nmse_db
from mf_csi.ema import EMA
from torch.optim.lr_scheduler import OneCycleLR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="runs/diu")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)         # legacy (unused when unet/lstm lr set)
    p.add_argument("--unet-lr", type=float, default=1e-3)    # paper: diffusion_lr
    p.add_argument("--lstm-lr", type=float, default=1e-4)    # paper: LSTM_lr
    p.add_argument("--ema-decay", type=float, default=0.95)  # paper: model_ema_decay
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def lr_at(step, cfg):
    """Linear warmup, then cosine decay to ~1% of peak (OneCycle-like)."""
    peak, warm, total = cfg.train.lr, cfg.train.warmup_steps, cfg.train.total_steps
    if warm > 0 and step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


def random_split_batch(past, future):
    """Concatenate history+future and pick a random split -> (history, target_frame).
    Mirrors the paper's random T_in single-next-frame training."""
    full = torch.cat([past, future], dim=1)          # [B, T, 2, Nt, Nc]
    T = full.shape[1]
    t_in = int(torch.randint(1, T, (1,)).item())     # history length in [1, T-1]
    return full[:, :t_in], full[:, t_in]


@torch.no_grad()
def evaluate(encoder, unet, scheduler, val_batches, cfg, device):
    per_step_sum = None
    snr = cfg.train.eval_snr_db
    for b in val_batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = ddim_ar_predict(encoder, unet, scheduler, hist, future.shape[1], cfg.diu)
        ps, _ = nmse(pred, future)
        per_step_sum = ps if per_step_sum is None else per_step_sum + ps
    per_step = per_step_sum / len(val_batches)
    return per_step, per_step.mean()


def save_ckpt(path, step, enc, unet, ema_enc, ema_unet, opt, best, global_ab=None):
    torch.save({"step": step, "best_nmse_db": best, "global_ab": global_ab,
                "enc": enc.state_dict(), "unet": unet.state_dict(),
                "ema_enc": ema_enc.state_dict(), "ema_unet": ema_unet.state_dict(),
                "opt": opt.state_dict()}, path)


def main():
    args = parse_args()
    cfg = Config()
    cfg.data.normalization = "global_minmax11"       # global [-1,1] min-max (matches the paper)
    if args.steps: cfg.train.total_steps = args.steps
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.lr: cfg.train.lr = args.lr
    if args.eval_every: cfg.train.eval_every = args.eval_every
    if args.ckpt_every: cfg.train.ckpt_every = args.ckpt_every
    cfg.train.out_dir = args.out_dir
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    torch.manual_seed(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} out_dir={cfg.train.out_dir} steps={cfg.train.total_steps} "
          f"batch={cfg.train.batch_size} lr={cfg.train.lr}")

    enc = DiUEncoder(cfg.diu, in_channels=2).to(device)
    unet = DiUNet(cfg.diu, data_channels=2, image_size=cfg.data.num_subcarriers_used).to(device)
    scheduler = make_scheduler(cfg.diu)
    huber = nn.HuberLoss(delta=cfg.diu.huber_delta)

    params = list(enc.parameters()) + list(unet.parameters())
    # Paper recipe: separate LR groups (U-Net 1e-3, ConvLSTM encoder 1e-4), Adam
    # default betas, no weight decay. (Under OneCycleLR with a scalar max_lr both
    # groups follow the same 1e-3 schedule -- matching the paper's code behaviour.)
    opt = torch.optim.Adam(unet.parameters(), lr=args.unet_lr)
    opt.add_param_group({"params": list(enc.parameters()), "lr": args.lstm_lr})
    # EMA decay 0.95, updated every 10 steps (paper recipe), evaluated as the model.
    EMA_EVERY = 10
    ema_enc, ema_unet = EMA(enc, args.ema_decay), EMA(unet, args.ema_decay)
    enc_eval = DiUEncoder(cfg.diu, in_channels=2).to(device)
    unet_eval = DiUNet(cfg.diu, data_channels=2, image_size=cfg.data.num_subcarriers_used).to(device)

    start_step, best = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        enc.load_state_dict(ck["enc"]); unet.load_state_dict(ck["unet"])
        ema_enc.load_state_dict(ck["ema_enc"]); ema_unet.load_state_dict(ck["ema_unet"])
        opt.load_state_dict(ck["opt"]); start_step = ck["step"] + 1
        best = ck.get("best_nmse_db", float("inf"))
        print(f"resumed from {args.resume} at step {start_step}")

    # Paper LR schedule: OneCycle (warmup 25%, cosine anneal) over the full budget.
    sched = OneCycleLR(opt, max_lr=args.unet_lr, total_steps=cfg.train.total_steps,
                       pct_start=0.25, anneal_strategy="cos",
                       last_epoch=(start_step - 1 if start_step > 0 else -1))

    # Global min-max (fit once), reused for train + eval so the space is consistent.
    global_ab = ck.get("global_ab") if (args.resume and os.path.isfile(args.resume)) else None
    if global_ab is None:
        print("estimating global min-max ...")
        global_ab = estimate_global_minmax(cfg.data, num_samples=4000, batch_size=256)
    print(f"global norm (a, b) = ({global_ab[0]:.4f}, {global_ab[1]:.4f})")

    print("building fixed validation set ...")
    val_batches = make_fixed_eval_set(cfg.data, cfg.train.val_samples,
                                      cfg.train.val_batch_size, global_ab=global_ab)
    train_iter = iter(DataLoader(
        CSIStreamDataset(cfg.data, batch_size=cfg.train.batch_size, steps_per_epoch=None,
                         global_ab=global_ab),
        batch_size=None))

    enc.train(); unet.train()
    t0 = time.time(); running, running_n = 0.0, 0
    for step in range(start_step, cfg.train.total_steps):
        batch = next(train_iter)
        past, future = batch["past"].to(device), batch["future"].to(device)
        history, target = random_split_batch(past, future)

        loss, m = diffusion_loss(enc, unet, scheduler, history, target, cfg.diu, huber)
        opt.zero_grad(); loss.backward()
        if cfg.train.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip_norm)
        opt.step()
        sched.step()
        if step % EMA_EVERY == 0:
            ema_enc.update(enc, args.ema_decay); ema_unet.update(unet, args.ema_decay)
        running += m["loss"].item(); running_n += 1

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:6d} | huber {running / running_n:.4f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {rate:.1f} it/s", flush=True)
            running, running_n = 0.0, 0

        if step > 0 and step % cfg.train.eval_every == 0:
            ema_enc.copy_to(enc_eval); ema_unet.copy_to(unet_eval)
            per_step, overall = evaluate(enc_eval, unet_eval, scheduler, val_batches, cfg, device)
            avg_db = nmse_db(overall).item()
            s1, sN = nmse_db(per_step[0]).item(), nmse_db(per_step[-1]).item()
            print(f"  [eval] step {step} | EMA avg NMSE {avg_db:.2f} dB "
                  f"(step1 {s1:.2f}, step{len(per_step)} {sN:.2f})", flush=True)
            if avg_db < best:
                best = avg_db
                save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_best.pt"),
                          step, enc, unet, ema_enc, ema_unet, opt, best, global_ab)

        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
                      step, enc, unet, ema_enc, ema_unet, opt, best, global_ab)

    save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
              cfg.train.total_steps - 1, enc, unet, ema_enc, ema_unet, opt, best, global_ab)
    print(f"done. best avg NMSE {best:.2f} dB. checkpoints in {cfg.train.out_dir}")


if __name__ == "__main__":
    main()
