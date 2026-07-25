"""Train the joint-horizon ConvLSTM regression baseline (JointRegressor).

Predicts all Nf future frames in ONE forward pass (no autoregressive rollout) and
is trained with plain MSE. Mirrors scripts/train.py's loop (warmup + cosine LR,
grad clip, EMA, periodic per-step NMSE eval, checkpoint/resume) but with the
regression objective. This is the rollout-free NMSE 'ceiling' baseline.

Usage (Alvis):
    python -m scripts.train_regression --out-dir runs/reg1 --steps 50000
"""

from __future__ import annotations

import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset, make_fixed_eval_set
from mf_csi.models import JointRegressor
from mf_csi.diffusion import corrupt_history          # additive scale-preserving noise
from mf_csi.inference import nmse, nmse_db
from mf_csi.ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="runs/reg")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    if args.steps is not None:       cfg.train.total_steps = args.steps
    if args.batch_size is not None:  cfg.train.batch_size = args.batch_size
    if args.lr is not None:          cfg.train.lr = args.lr
    if args.eval_every is not None:  cfg.train.eval_every = args.eval_every
    if args.ckpt_every is not None:  cfg.train.ckpt_every = args.ckpt_every
    if args.seed is not None:        cfg.train.seed = args.seed
    cfg.train.out_dir = args.out_dir
    return cfg


def lr_at(step, cfg):
    peak, warm, total = cfg.train.lr, cfg.train.warmup_steps, cfg.train.total_steps
    if warm > 0 and step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


def augment(past, cfg):
    if not cfg.regression.noise_aug:
        return past
    return corrupt_history(past, cfg.regression.snr_db_min, cfg.regression.snr_db_max)


@torch.no_grad()
def evaluate(model, val_batches, cfg, device):
    per_step_sum = None
    snr = cfg.train.eval_snr_db
    for b in val_batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = model(hist)                                  # [B, Nf, 2, Nt, Nc] one shot
        ps, _ = nmse(pred, future)
        per_step_sum = ps if per_step_sum is None else per_step_sum + ps
    per_step = per_step_sum / len(val_batches)
    return per_step, per_step.mean()


def save_ckpt(path, step, model, ema, opt, best):
    torch.save({"step": step, "best_nmse_db": best,
                "model": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict()}, path)


def main():
    args = parse_args()
    cfg = build_config(args)
    cfg.data.normalization = "std"                          # matches MeanFlow / physical-space eval
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    torch.manual_seed(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} out_dir={cfg.train.out_dir} steps={cfg.train.total_steps} "
          f"batch={cfg.train.batch_size} lr={cfg.train.lr} (JointRegressor, joint {cfg.data.num_future}-frame)")

    model = JointRegressor(cfg.regression).to(device)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=cfg.train.lr,
                           betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
                           weight_decay=cfg.train.weight_decay)
    ema = EMA(model, cfg.train.ema_decay)
    model_eval = JointRegressor(cfg.regression).to(device)

    start_step, best = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); start_step = ck["step"] + 1
        best = ck.get("best_nmse_db", float("inf"))
        print(f"resumed from {args.resume} at step {start_step}")

    print("building fixed validation set ...")
    val_batches = make_fixed_eval_set(cfg.data, cfg.train.val_samples, cfg.train.val_batch_size)
    train_iter = iter(DataLoader(
        CSIStreamDataset(cfg.data, batch_size=cfg.train.batch_size, steps_per_epoch=None),
        batch_size=None))

    model.train()
    t0 = time.time(); running, running_n = 0.0, 0
    for step in range(start_step, cfg.train.total_steps):
        batch = next(train_iter)
        past, future = batch["past"].to(device), batch["future"].to(device)
        hist = augment(past, cfg)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        pred = model(hist)
        loss = F.mse_loss(pred, future)
        opt.zero_grad(); loss.backward()
        if cfg.train.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip_norm)
        opt.step()
        eff = min(cfg.train.ema_decay, (1.0 + step) / (10.0 + step))
        ema.update(model, eff)
        running += loss.item(); running_n += 1

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:6d} | mse {running / running_n:.4f} "
                  f"| lr {lr_at(step, cfg):.2e} | {rate:.1f} it/s", flush=True)
            running, running_n = 0.0, 0

        if step > 0 and step % cfg.train.eval_every == 0:
            ema.copy_to(model_eval)
            per_step, overall = evaluate(model_eval, val_batches, cfg, device)
            _, overall_raw = evaluate(model, val_batches, cfg, device)
            avg_db = nmse_db(overall).item(); raw_db = nmse_db(overall_raw).item()
            s1, sN = nmse_db(per_step[0]).item(), nmse_db(per_step[-1]).item()
            print(f"  [eval] step {step} | EMA avg NMSE {avg_db:.2f} dB "
                  f"(step1 {s1:.2f}, step{len(per_step)} {sN:.2f}) | raw {raw_db:.2f} dB", flush=True)
            if avg_db < best:
                best = avg_db
                save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_best.pt"), step, model, ema, opt, best)

        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"), step, model, ema, opt, best)

    save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
              cfg.train.total_steps - 1, model, ema, opt, best)
    print(f"done. best avg NMSE {best:.2f} dB. checkpoints in {cfg.train.out_dir}")


if __name__ == "__main__":
    main()
