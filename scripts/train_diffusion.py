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
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset, make_fixed_eval_set
from mf_csi.models.diu import DiUEncoder, DiUNet
from mf_csi.diffusion import make_scheduler, diffusion_loss, ddim_ar_predict
from mf_csi.inference import nmse, nmse_db
from mf_csi.ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="runs/diu")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def lr_at(step, cfg):
    if cfg.train.warmup_steps > 0 and step < cfg.train.warmup_steps:
        return cfg.train.lr * (step + 1) / cfg.train.warmup_steps
    return cfg.train.lr


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
    for b in val_batches:
        past, future = b["past"].to(device), b["future"].to(device)
        pred = ddim_ar_predict(encoder, unet, scheduler, past, future.shape[1], cfg.diu)
        ps, _ = nmse(pred, future)
        per_step_sum = ps if per_step_sum is None else per_step_sum + ps
    per_step = per_step_sum / len(val_batches)
    return per_step, per_step.mean()


def save_ckpt(path, step, enc, unet, ema_enc, ema_unet, opt, best):
    torch.save({"step": step, "best_nmse_db": best,
                "enc": enc.state_dict(), "unet": unet.state_dict(),
                "ema_enc": ema_enc.state_dict(), "ema_unet": ema_unet.state_dict(),
                "opt": opt.state_dict()}, path)


def main():
    args = parse_args()
    cfg = Config()
    cfg.data.normalization = "minmax11"              # diffusion works in [-1, 1]
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
    opt = torch.optim.Adam(params, lr=cfg.train.lr,
                           betas=(cfg.train.adam_beta1, cfg.train.adam_beta2))
    ema_enc, ema_unet = EMA(enc, cfg.train.ema_decay), EMA(unet, cfg.train.ema_decay)
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

    print("building fixed validation set ...")
    val_batches = make_fixed_eval_set(cfg.data, cfg.train.val_samples, cfg.train.val_batch_size)
    train_iter = iter(DataLoader(
        CSIStreamDataset(cfg.data, batch_size=cfg.train.batch_size, steps_per_epoch=None),
        batch_size=None))

    enc.train(); unet.train()
    t0 = time.time(); running, running_n = 0.0, 0
    for step in range(start_step, cfg.train.total_steps):
        batch = next(train_iter)
        past, future = batch["past"].to(device), batch["future"].to(device)
        history, target = random_split_batch(past, future)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        loss, m = diffusion_loss(enc, unet, scheduler, history, target, cfg.diu, huber)
        opt.zero_grad(); loss.backward()
        if cfg.train.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip_norm)
        opt.step()
        eff = min(cfg.train.ema_decay, (1.0 + step) / (10.0 + step))
        ema_enc.update(enc, eff); ema_unet.update(unet, eff)
        running += m["loss"].item(); running_n += 1

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:6d} | huber {running / running_n:.4f} "
                  f"| lr {lr_at(step, cfg):.2e} | {rate:.1f} it/s", flush=True)
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
                          step, enc, unet, ema_enc, ema_unet, opt, best)

        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
                      step, enc, unet, ema_enc, ema_unet, opt, best)

    save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
              cfg.train.total_steps - 1, enc, unet, ema_enc, ema_unet, opt, best)
    print(f"done. best avg NMSE {best:.2f} dB. checkpoints in {cfg.train.out_dir}")


if __name__ == "__main__":
    main()
