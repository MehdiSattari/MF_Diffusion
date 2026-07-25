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
import math
import os
import time
import torch
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset, make_fixed_eval_set
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.meanflow import meanflow_loss
from mf_csi.inference import autoregressive_predict, nmse, nmse_db
from mf_csi.diffusion import corrupt_history
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
    p.add_argument("--source-std", type=float, default=None,
                   help="sigma of the flow source eps~N(0,sigma^2); small (0.1-0.3) "
                        "= near-deterministic mean-seeking 1-NFE output")
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
    if args.source_std is not None:  cfg.meanflow.source_std = args.source_std
    # Inference seed scale MUST match the training source scale.
    cfg.inference.seed_std = cfg.meanflow.source_std
    return cfg


def lr_at(step: int, cfg) -> float:
    """Linear warmup, then cosine decay to ~1% of peak (matches the diffusion run)."""
    peak, warm, total = cfg.train.lr, cfg.train.warmup_steps, cfg.train.total_steps
    if warm > 0 and step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


@torch.no_grad()
def evaluate(enc_eval, gen_eval, val_batches, cfg, device):
    per_step_sum = None
    snr = cfg.train.eval_snr_db
    for b in val_batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = autoregressive_predict(
            enc_eval, gen_eval, hist, future.shape[1],
            seed_std=cfg.inference.seed_std, step_noise_std=cfg.inference.step_noise_std,
            num_samples=cfg.inference.mean_samples)
        ps, _ = nmse(pred, future)
        per_step_sum = ps if per_step_sum is None else per_step_sum + ps
    per_step = per_step_sum / len(val_batches)
    return per_step, per_step.mean()


def save_ckpt(path, step, enc, gen, ema_enc, ema_gen, opt, best, source_std):
    torch.save({
        "step": step, "best_nmse_db": best, "source_std": source_std,
        "enc": enc.state_dict(), "gen": gen.state_dict(),
        "ema_enc": ema_enc.state_dict(), "ema_gen": ema_gen.state_dict(),
        "opt": opt.state_dict(),
    }, path)


def main():
    args = parse_args()
    cfg = build_config(args)
    cfg.data.normalization = "std"           # per-sample zero-mean/unit-std: makes the
                                             # wide sigma=1 informative-prior scale-matched
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    torch.manual_seed(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} out_dir={cfg.train.out_dir} steps={cfg.train.total_steps} "
          f"batch={cfg.train.batch_size} lr={cfg.train.lr} "
          f"source_std={cfg.meanflow.source_std} mean_samples={cfg.inference.mean_samples}")

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
    running, running_mu, running_n = 0.0, 0.0, 0
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
        # EMA warmup: track fast early, ramp toward the configured decay so the
        # EMA weights are meaningful long before ~1/(1-decay) steps have passed.
        eff_decay = min(cfg.train.ema_decay, (1.0 + step) / (10.0 + step))
        ema_enc.update(enc, eff_decay); ema_gen.update(gen, eff_decay)
        running += m["mse"].item(); running_mu += m.get("mu_mse", 0.0*m["mse"]).item(); running_n += 1

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:6d} | mse {running / running_n:.4f} "
                  f"| mu_mse {running_mu / running_n:.4f} "
                  f"| lr {lr_at(step, cfg):.2e} | {rate:.1f} it/s", flush=True)
            running, running_mu, running_n = 0.0, 0.0, 0

        if step > 0 and step % cfg.train.eval_every == 0:
            ema_enc.copy_to(enc_eval); ema_gen.copy_to(gen_eval)
            per_step, overall = evaluate(enc_eval, gen_eval, val_batches, cfg, device)
            _, overall_raw = evaluate(enc, gen, val_batches, cfg, device)   # raw weights too
            avg_db = nmse_db(overall).item()
            raw_db = nmse_db(overall_raw).item()
            s1, sN = nmse_db(per_step[0]).item(), nmse_db(per_step[-1]).item()
            print(f"  [eval] step {step} | EMA avg NMSE {avg_db:.2f} dB "
                  f"(step1 {s1:.2f}, step{len(per_step)} {sN:.2f}) | raw {raw_db:.2f} dB",
                  flush=True)
            if avg_db < best:
                best = avg_db
                save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_best.pt"),
                          step, enc, gen, ema_enc, ema_gen, opt, best,
                          cfg.meanflow.source_std)

        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
                      step, enc, gen, ema_enc, ema_gen, opt, best,
                      cfg.meanflow.source_std)

    save_ckpt(os.path.join(cfg.train.out_dir, "ckpt_last.pt"),
              cfg.train.total_steps - 1, enc, gen, ema_enc, ema_gen, opt, best,
              cfg.meanflow.source_std)
    print(f"done. best avg NMSE {best:.2f} dB. checkpoints in {cfg.train.out_dir}")


if __name__ == "__main__":
    main()
