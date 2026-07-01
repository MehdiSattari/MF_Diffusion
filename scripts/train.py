"""MeanFlow DiU training loop for CSI prediction.

Trains the ConvLSTM temporal encoder + U-Net generator with the MeanFlow
objective, using gradient clipping, LR warmup, and EMA weights. Periodically
evaluates autoregressive NMSE on a fixed validation set (EMA weights) and
checkpoints with resume support.

Usage (on Alvis, inside the venv):
    python -m scripts.train --out-dir runs/exp1 --steps 50000
    python -m scripts.train --out-dir runs/exp1 --resume runs/exp1/ckpt_last.pt
"""

from __future__ import annotations

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset, make_fixed_eval_set
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.meanflow import meanflow_loss
from mf_csi.inference import autoregressive_predict, nmse, nmse_db
from mf_csi.ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=None)
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
    if args.out_dir is not None:     cfg.train.out_dir = args.out_dir
    if args.steps is not None:       cfg.train.total_steps = args.steps
    if args.batch_size is not None:  cfg.train.batch_size = args.batch_size
    if args.lr is not None:          cfg.train.lr = args.lr
    if args.eval_every is not None:  cfg.train.eval_every = args.eval_every
    if args.ckpt_every is not None:  cfg.train.ckpt_every = args.ckpt_every
    if args.seed is not None:        cfg.train.seed = args.seed
    return cfg


def lr_at(step: int, cfg) -> float:
    if cfg.train.warmup_steps > 0 and step < cfg.train.warmup_steps:
        return cfg.train.lr * (step + 1) / cfg.train.warmup_steps
    return cfg.train.lr


@torch.no_grad()
def evaluate(enc_eval, gen_eval, val_batches, cfg, device):
    per_step_sum = None
    for b in val_batches:
        past, future = b["past"].to(device), b["future"].to(device)
        pred = autoregressive_predict(
            enc_eval, gen_eval, past, future.shape[1],
            seed_std=cfg.inference.seed_std, step_noise_std=cfg.inference.step_noise_std)
        ps, _ = nmse(pred, future)
        per_step_sum = ps if per_step_sum is None else per_step_sum + ps
    per_step = per_step_sum / len(val_batches)
    return per_step, per_step.mean()


def save_ckpt(path, step, enc, gen, ema_enc, ema_gen, opt, best):
    torch.save({
        "step": step, "best_nmse_db": best,
        "enc": enc.state_dict(), "gen": gen.state_dict(),
        "ema_enc": ema_enc.state_dict(), "ema_gen": ema_gen.state_dict(),
        "opt": opt.state_dict(),
    }, path)


def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    torch.manual_seed(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} out_dir={cfg.train.out_dir} steps={cfg.train.total_steps} "
          f"batch={cfg.train.batch_size} lr={cfg.train.lr}")

    enc = TemporalEncoder(cfg.encoder).to(device)
    gen = UNetGenerator(cfg.generator).to(device)
    params = list(enc.parameters()) + list(gen.parameters())
    opt = torch.optim.Adam(params, lr=cfg.train.lr,
                           betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
                           weight_decay=cfg.train.weight_decay)
    ema_enc = EMA(enc, cfg.train.ema_decay)
    ema_gen = EMA(gen, cfg.train.ema_decay)

    # Eval-only modules that receive the EMA weights before each evaluation.
    enc_eval = TemporalEncoder(cfg.encoder).to(device)
    gen_eval = UNetGenerator(cfg.generator).to(device)

    start_step, best = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        enc.load_state_dict(ck["enc"]); gen.load_state_dict(ck["gen"])
        ema_enc.load_state_dict(ck["ema_enc"]); ema_gen.load_state_dict(ck["ema_gen"])
        opt.load_state_dict(ck["opt"]); start_step = ck["step"] + 1
        best = ck.get("best_nmse_db", float("inf"))
        print(f"resumed from {args.resume} at step {start_step}")

    # Fixed validation set (generated once, reproducible).
    print("building fixed validation set ...")
    val_batches = make_fixed_eval_set(cfg.data, cfg.train.val_samples, cfg.train.val_batch_size)

    train_ds = CSIStreamDataset(cfg.data, batch_size=cfg.train.batch_size, steps_per_epoch=None)
    train_iter = iter(DataLoader(train_ds, batch_size=None))

    enc.train(); gen.train()
    t0 = time.time()
    running, running_n = 0.0, 0
    for step in range(start_step, cfg.train.total_steps):
        batch = next(train_iter)
        past, future = batch["past"].to(device), batch["future"].to(device)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        loss, m = meanflow_loss(enc, gen, past, future, cfg.meanflow)
        opt.zero_grad()
        loss.backward()
        if cfg.train.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip_norm)
        opt.step()
        ema_enc.update(enc); ema_gen.update(gen)
        running += m["mse"].item(); running_n += 1

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:6d} | mse {running / running_n:.4f} "
                  f"| lr {lr_at(step, cfg):.2e} | {rate:.1f} it/s", flush=True)
            running, running_n = 0.0, 0

        if step > 0 and step % cfg.train.eval_every == 0:
            ema_enc.copy_to(enc_eval); ema_gen.copy_to(gen_eval)
            per_step, overall = evaluate(enc_eval, gen_eval, val_batches, cfg, device)
            avg_db = nmse_db(overall).item()
            s1, sN = nmse_db(per_step[0]).item(), nmse_db(per_step[-1]).item()
            print(f"  [eval] step {step} | avg NMSE {avg_db:.2f} dB "
                  f"(step1 {s1:.2f}, step{len(per_step)} {sN:.2f})", flush=True)
            if avg_db < best:
                best = avg_db
                save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_best.pt"),
                          step, enc, gen, ema_enc, ema_gen, opt, best)

        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
                      step, enc, gen, ema_enc, ema_gen, opt, best)

    save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
              cfg.train.total_steps - 1, enc, gen, ema_enc, ema_gen, opt, best)
    print(f"done. best avg NMSE {best:.2f} dB. checkpoints in {cfg.train.out_dir}")


if __name__ == "__main__":
    main()
