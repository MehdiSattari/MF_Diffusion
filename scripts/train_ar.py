"""Unified AUTOREGRESSIVE trainer for the controlled 2x2 mu-ablation.

One script, one shared encoder + backbone (TemporalEncoder + UNetGenerator), AR
inference. Only --objective and --mu change:

    --objective meanflow  : average-velocity, 1-NFE sampling
    --objective diffusion : predict-x0/residual, DDIM (--diff-steps, default 3 = paper)
    --mu on               : informative prior (MeanFlow: source centered on mu;
                            diffusion: residual diffusion around mu) + aux MSE(mu,Y)
    --mu off              : plain prior (source ~ N(0,I) / standard diffusion)

Everything else is shared (global/std norm, Adam + cosine-warmup, EMA, grad clip,
same data stream), so a MeanFlow-vs-diffusion or mu-vs-no-mu difference is attributable
only to those switches. Fairness: MeanFlow 1 NFE <= diffusion 3 NFE, same backbone.

Usage (Alvis):
    python -m scripts.train_ar --objective meanflow  --mu off --out-dir runs/ar_mf_nomu
    python -m scripts.train_ar --objective diffusion --mu on  --out-dir runs/ar_di_mu
    python -m scripts.train_ar --objective diffusion --mu off --out-dir runs/ar_di_nomu
"""

from __future__ import annotations

import argparse, math, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset, make_fixed_eval_set
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.meanflow import meanflow_loss
from mf_csi.diffusion import make_scheduler, corrupt_history
from mf_csi.diffusion_shared import diffusion_loss_shared, ddim_ar_predict_shared
from mf_csi.inference import autoregressive_predict, nmse, nmse_db
from mf_csi.ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["meanflow", "diffusion"])
    p.add_argument("--mu", required=True, choices=["on", "off"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--diff-steps", type=int, default=3, help="DDIM steps (diffusion eval)")
    p.add_argument("--prediction-type", type=str, default=None,
                   choices=["sample", "epsilon", "v_prediction"],
                   help="diffusion parameterization; default keeps config's 'sample' (predict-x0). "
                        "'v_prediction' gives the monotone NMSE-improves-with-steps behaviour.")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-psd", type=str, default=None,
                   help="path to a residual PSD (.pt) for the channel-shaped colored source (MeanFlow)")
    p.add_argument("--gen-size", type=str, default=None,
                   choices=["xs", "small", "medium", "large", "xl"],
                   help="scale the shared generator for the accuracy-complexity Pareto")
    return p.parse_args()


def lr_at(step, peak, warm, total):
    if warm > 0 and step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


@torch.no_grad()
def evaluate(enc, gen, val_batches, cfg, device, objective, scheduler, use_mu):
    snr = cfg.train.eval_snr_db
    ps_sum = None
    for b in val_batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        Nf = future.shape[1]
        if objective == "meanflow":
            pred = autoregressive_predict(enc, gen, hist, Nf, seed_std=cfg.inference.seed_std,
                                          num_samples=cfg.inference.mean_samples)
        else:
            pred = ddim_ar_predict_shared(enc, gen, scheduler, hist, Nf, cfg.diu, use_mu=use_mu)
        ps, _ = nmse(pred, future)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    ps = ps_sum / len(val_batches)
    return ps, ps.mean()


def save_ckpt(path, step, enc, gen, ema_enc, ema_gen, opt, best, meta):
    torch.save({"step": step, "best_nmse_db": best, "meta": meta,
                "enc": enc.state_dict(), "gen": gen.state_dict(),
                "ema_enc": ema_enc.state_dict(), "ema_gen": ema_gen.state_dict(),
                "opt": opt.state_dict()}, path)


def main():
    args = parse_args()
    use_mu = (args.mu == "on")
    cfg = Config()
    cfg.data.normalization = "std"
    if args.gen_size:
        from mf_csi.config import apply_generator_size
        apply_generator_size(cfg.generator, args.gen_size)
        print(f"generator size={args.gen_size}: base_ch={cfg.generator.base_channels} "
              f"res_blocks={cfg.generator.num_res_blocks} attn={cfg.generator.use_attention}", flush=True)
    cfg.train.total_steps = args.steps
    cfg.train.batch_size = args.batch_size
    cfg.train.eval_every = args.eval_every
    cfg.train.ckpt_every = args.ckpt_every
    cfg.diu.sampling_steps = args.diff_steps
    if args.prediction_type:
        cfg.diu.prediction_type = args.prediction_type
        print(f"diffusion prediction_type={cfg.diu.prediction_type}", flush=True)
    cfg.meanflow.informative_prior = use_mu
    cfg.inference.seed_std = cfg.meanflow.source_std
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_psd = None
    if args.source_psd:
        source_psd = torch.load(args.source_psd, map_location=device)["psd"].to(device)
        print(f"colored source: loaded residual PSD from {args.source_psd}", flush=True)

    enc = TemporalEncoder(cfg.encoder).to(device)
    gen = UNetGenerator(cfg.generator).to(device)
    enc_eval = TemporalEncoder(cfg.encoder).to(device)
    gen_eval = UNetGenerator(cfg.generator).to(device)
    scheduler = make_scheduler(cfg.diu)
    huber = nn.HuberLoss(delta=cfg.diu.huber_delta)
    np_ = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in gen.parameters())
    print(f"objective={args.objective} mu={args.mu} params={np_/1e6:.3f}M steps={args.steps} "
          f"diff_steps={args.diff_steps} norm=std", flush=True)

    params = list(enc.parameters()) + list(gen.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, betas=(cfg.train.adam_beta1, cfg.train.adam_beta2))
    ema_enc, ema_gen = EMA(enc, cfg.train.ema_decay), EMA(gen, cfg.train.ema_decay)

    start_step, best = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        enc.load_state_dict(ck["enc"]); gen.load_state_dict(ck["gen"])
        ema_enc.load_state_dict(ck["ema_enc"]); ema_gen.load_state_dict(ck["ema_gen"])
        opt.load_state_dict(ck["opt"]); start_step = ck["step"] + 1
        best = ck.get("best_nmse_db", float("inf"))
        print(f"resumed at {start_step}")

    val_batches = make_fixed_eval_set(cfg.data, cfg.train.val_samples, cfg.train.val_batch_size)
    train_iter = iter(DataLoader(
        CSIStreamDataset(cfg.data, batch_size=cfg.train.batch_size, steps_per_epoch=None), batch_size=None))
    meta = {"objective": args.objective, "mu": args.mu, "diff_steps": args.diff_steps,
            "source_psd": args.source_psd, "prediction_type": cfg.diu.prediction_type,
            "gen_size": args.gen_size}

    enc.train(); gen.train()
    t0 = time.time(); running, running_aux, running_n = 0.0, 0.0, 0
    for step in range(start_step, cfg.train.total_steps):
        batch = next(train_iter)
        past, future = batch["past"].to(device), batch["future"].to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, cfg.train.warmup_steps, cfg.train.total_steps)
        if args.objective == "meanflow":
            loss, m = meanflow_loss(enc, gen, past, future, cfg.meanflow, source_psd=source_psd)
            aux = m.get("mu_mse", torch.zeros(()))
        else:
            loss, m = diffusion_loss_shared(enc, gen, scheduler, past, future[:, 0], cfg.diu, huber,
                                            use_mu=use_mu)
            aux = m.get("aux", torch.zeros(()))
        opt.zero_grad(); loss.backward()
        if cfg.train.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip_norm)
        opt.step()
        eff = min(cfg.train.ema_decay, (1.0 + step) / (10.0 + step))
        ema_enc.update(enc, eff); ema_gen.update(gen, eff)
        running += m["loss"].item(); running_aux += float(aux); running_n += 1

        if step % cfg.train.log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            print(f"step {step:6d} | loss {running/running_n:.4f} | aux {running_aux/running_n:.4f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {rate:.1f} it/s", flush=True)
            running, running_aux, running_n = 0.0, 0.0, 0
        if step > 0 and step % cfg.train.eval_every == 0:
            ema_enc.copy_to(enc_eval); ema_gen.copy_to(gen_eval)
            per_step, overall = evaluate(enc_eval, gen_eval, val_batches, cfg, device,
                                         args.objective, scheduler, use_mu)
            avg_db = nmse_db(overall).item()
            print(f"  [eval] step {step} | EMA avg NMSE {avg_db:.2f} dB "
                  f"(step1 {nmse_db(per_step[0]).item():.2f}, step{len(per_step)} {nmse_db(per_step[-1]).item():.2f})",
                  flush=True)
            if avg_db < best:
                best = avg_db
                save_ckpt(os.path.join(args.out_dir, "ckpt_best.pt"), step, enc, gen, ema_enc, ema_gen, opt, best, meta)
        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_ckpt(os.path.join(args.out_dir, "ckpt_last.pt"), step, enc, gen, ema_enc, ema_gen, opt, best, meta)

    save_ckpt(os.path.join(args.out_dir, "ckpt_last.pt"), cfg.train.total_steps - 1,
              enc, gen, ema_enc, ema_gen, opt, best, meta)
    print(f"done. best avg NMSE {best:.2f} dB. in {args.out_dir}")


if __name__ == "__main__":
    main()
