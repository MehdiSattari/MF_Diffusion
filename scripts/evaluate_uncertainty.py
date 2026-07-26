"""Uncertainty + downstream evaluation of the generative CSI predictors.

Draws K samples per input from the generative models (MeanFlow; optionally stochastic
DiU), and reports -- in the PHYSICAL channel space -- the metrics that reveal whether
the generative residual adds value beyond the conditional mean:

    ensemble-mean NMSE, CRPS, coverage/reliability, spread-skill, spectral efficiency.

The regressor (JointRegressor) is a deterministic point baseline: it gets NMSE and SE
but, by construction, no calibrated uncertainty -- which is exactly the point.

Usage (Alvis):
    python -m scripts.evaluate_uncertainty --K 30 --snr 20 \
        --mf-ckpt runs/mf_6913678/ckpt_best.pt \
        --reg-ckpt runs/reg_6915230/ckpt_best.pt \
        --diu-ckpt runs/diu_6793522/ckpt_best.pt
"""

from __future__ import annotations

import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mf_csi.config import Config
from mf_csi.data.sionna_cdl import CDLChannelGenerator
from mf_csi.data.dataset import _normalize, _split_batch, estimate_global_minmax, denormalize
from mf_csi.models import TemporalEncoder, UNetGenerator, JointRegressor
from mf_csi.models.diu import DiUEncoder, DiUNet
from mf_csi.diffusion import make_scheduler, ddim_ar_predict, corrupt_history
from mf_csi.inference import autoregressive_predict, nmse, nmse_db
from mf_csi.uncertainty import (crps, coverage, spread_skill, ensemble_mean_nmse,
                                spectral_efficiency, outage_rate)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mf-ckpt", type=str, default=None)
    p.add_argument("--diu-ckpt", type=str, default=None)
    p.add_argument("--reg-ckpt", type=str, default=None)
    p.add_argument("--K", type=int, default=30, help="ensemble size (generative samples)")
    p.add_argument("--snr", type=float, default=20.0, help="inference SNR (history corruption)")
    p.add_argument("--n-samples", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--step-noise", type=float, default=0.0,
                   help="per-step rollout noise -> grows late-horizon spread (calibration)")
    p.add_argument("--epsilon", type=float, default=0.1, help="target outage for risk-aware rate")
    p.add_argument("--out-dir", type=str, default="runs/uq")
    return p.parse_args()


def raw_batches(cfg, n, bs):
    gen = CDLChannelGenerator(cfg); out = []; rem = n
    while rem > 0:
        b = min(bs, rem); out.append(gen.generate(b)); rem -= b
    return out


def to_batches(raw, cfg, mode, global_ab):
    out = []
    for c in raw:
        c_n, stats = _normalize(c, mode, global_ab)
        out.append(_split_batch(c_n, cfg, stats))
    return out


def denorm_ensemble(samples, stats):
    """samples [K,B,Nf,2,Nt,Nc] -> physical, denormalized per K."""
    return torch.stack([denormalize(samples[k], stats) for k in range(samples.shape[0])], dim=0)


@torch.no_grad()
def mf_samples(enc, gen, hist, Nf, seed_std, K, step_noise=0.0):
    out = [autoregressive_predict(enc, gen, hist, Nf, seed_std=seed_std,
                                  step_noise_std=step_noise, num_samples=1) for _ in range(K)]
    return torch.stack(out, dim=0)                       # [K,B,Nf,2,Nt,Nc]


@torch.no_grad()
def diu_samples(enc, unet, scheduler, hist, Nf, cfg, K):
    cfg.diu.deterministic_init = False                  # stochastic init for a real ensemble
    cfg.diu.ddim_eta = 1.0
    out = [ddim_ar_predict(enc, unet, scheduler, hist, Nf, cfg.diu) for _ in range(K)]
    return torch.stack(out, dim=0)


def accumulate(metrics, key, val):
    metrics.setdefault(key, []).append(val)


def main():
    args = parse_args()
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"generating shared eval set: {args.n_samples} samples | K={args.K} | SNR={args.snr}")
    raw = raw_batches(cfg.data, args.n_samples, args.batch_size)

    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}

    # ---------------- MeanFlow (generative) ----------------
    if args.mf_ckpt:
        ck = torch.load(args.mf_ckpt, map_location=device)
        enc = TemporalEncoder(cfg.encoder).to(device); gen = UNetGenerator(cfg.generator).to(device)
        enc.load_state_dict(ck["ema_enc"]); gen.load_state_dict(ck["ema_gen"]); enc.eval(); gen.eval()
        seed_std = ck.get("source_std", 1.0)
        batches = to_batches(raw, cfg.data, "std", None)
        agg = {}
        for b in batches:
            past, future = b["past"].to(device), b["future"].to(device)
            hist = corrupt_history(past, args.snr, args.snr)
            s = denorm_ensemble(mf_samples(enc, gen, hist, future.shape[1], seed_std, args.K, args.step_noise), b["stats"])
            y = denormalize(future, b["stats"])
            accumulate(agg, "nmse", ensemble_mean_nmse(s, y)[0])
            accumulate(agg, "crps", crps(s, y)[0])
            accumulate(agg, "spread", spread_skill(s, y)[0])
            accumulate(agg, "skill", spread_skill(s, y)[1])
            accumulate(agg, "se_pred", spectral_efficiency(s.mean(0), y, args.snr)[0])
            accumulate(agg, "se_perfect", spectral_efficiency(y, y, args.snr)[1])
            g_o, o_o = outage_rate(s, y, args.snr, args.epsilon)
            accumulate(agg, "goodput", g_o); accumulate(agg, "outage", o_o)
            for lv in levels:
                accumulate(agg, f"cov{lv}", coverage(s, y, lv)[1].item())
        results["MeanFlow"] = {k: (float(np.mean(v)) if k.startswith("cov")
                                   else torch.stack(v).mean(0).tolist()) for k, v in agg.items()}
        print(f"[MeanFlow] mean-NMSE {nmse_db(torch.tensor(results['MeanFlow']['nmse']).mean()).item():.2f} dB "
              f"| cov0.9 {results['MeanFlow']['cov0.9']:.3f} | CRPS {np.mean(results['MeanFlow']['crps']):.4f}")

    # ---------------- DiU (stochastic diffusion) ----------------
    if args.diu_ckpt:
        ck = torch.load(args.diu_ckpt, map_location=device)
        enc = DiUEncoder(cfg.diu, in_channels=2).to(device)
        unet = DiUNet(cfg.diu, data_channels=2, image_size=cfg.data.num_subcarriers_used).to(device)
        enc.load_state_dict(ck["ema_enc"]); unet.load_state_dict(ck["ema_unet"]); enc.eval(); unet.eval()
        gab = ck.get("global_ab") or estimate_global_minmax(cfg.data, 4000, 256)
        scheduler = make_scheduler(cfg.diu)
        batches = to_batches(raw, cfg.data, "global_minmax11", gab)
        agg = {}
        for b in batches:
            past, future = b["past"].to(device), b["future"].to(device)
            hist = corrupt_history(past, args.snr, args.snr)
            s = denorm_ensemble(diu_samples(enc, unet, scheduler, hist, future.shape[1], cfg, args.K), b["stats"])
            y = denormalize(future, b["stats"])
            accumulate(agg, "nmse", ensemble_mean_nmse(s, y)[0]); accumulate(agg, "crps", crps(s, y)[0])
            accumulate(agg, "spread", spread_skill(s, y)[0]); accumulate(agg, "skill", spread_skill(s, y)[1])
            accumulate(agg, "se_pred", spectral_efficiency(s.mean(0), y, args.snr)[0])
            g_o, o_o = outage_rate(s, y, args.snr, args.epsilon)
            accumulate(agg, "goodput", g_o); accumulate(agg, "outage", o_o)
            for lv in levels:
                accumulate(agg, f"cov{lv}", coverage(s, y, lv)[1].item())
        results["DiU"] = {k: (float(np.mean(v)) if k.startswith("cov")
                              else torch.stack(v).mean(0).tolist()) for k, v in agg.items()}
        print(f"[DiU] cov0.9 {results['DiU']['cov0.9']:.3f} | CRPS {np.mean(results['DiU']['crps']):.4f}")

    # ---------------- Regressor (point baseline; NMSE + SE only) ----------------
    if args.reg_ckpt:
        ck = torch.load(args.reg_ckpt, map_location=device)
        reg = JointRegressor(cfg.regression).to(device)
        reg.load_state_dict(ck.get("ema", ck.get("model"))); reg.eval()
        batches = to_batches(raw, cfg.data, "std", None)
        agg = {}
        for b in batches:
            past, future = b["past"].to(device), b["future"].to(device)
            hist = corrupt_history(past, args.snr, args.snr)
            pred = denormalize(reg(hist), b["stats"]); y = denormalize(future, b["stats"])
            s = pred.unsqueeze(0)                              # K=1 degenerate ensemble (a point)
            accumulate(agg, "nmse", ensemble_mean_nmse(s, y)[0])   # now vs y (bug fixed)
            accumulate(agg, "se_pred", spectral_efficiency(pred, y, args.snr)[0])
            g_o, o_o = outage_rate(s, y, args.snr, args.epsilon)
            accumulate(agg, "goodput", g_o); accumulate(agg, "outage", o_o)
            for lv in levels:                                 # coverage of a point ~ 0 (the contrast)
                accumulate(agg, f"cov{lv}", coverage(s, y, lv)[1].item())
        results["ConvLSTM"] = {k: (float(np.mean(v)) if k.startswith("cov")
                                   else torch.stack(v).mean(0).tolist()) for k, v in agg.items()}
        print(f"[ConvLSTM] point NMSE {nmse_db(torch.tensor(results['ConvLSTM']['nmse']).mean()).item():.2f} dB "
              f"| cov0.9 {results['ConvLSTM']['cov0.9']:.3f} (point -> ~0)")

    # ---------------- figures ----------------
    steps = list(range(1, cfg.data.num_future + 1))
    # reliability
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="ideal")
    for name in ("MeanFlow", "DiU", "ConvLSTM"):
        if name in results and f"cov{levels[0]}" in results[name]:
            emp = [results[name][f"cov{lv}"] for lv in levels]
            ax.plot(levels, emp, marker="o", label=name)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "reliability.png"), dpi=150); plt.close(fig)
    # spread-skill + SE
    for metric, ylab, title, fname in [("spread", "spread / skill", "Spread-skill (MeanFlow)", "spread_skill.png"),
                                       ("se_pred", "SE (b/s/Hz)", "Spectral efficiency", "spectral_efficiency.png")]:
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        if metric == "spread" and "MeanFlow" in results:
            ax.plot(steps, results["MeanFlow"]["spread"], marker="o", label="spread (ensemble std)")
            ax.plot(steps, results["MeanFlow"]["skill"], marker="s", label="skill (RMSE)")
        else:
            for name in ("MeanFlow", "DiU", "ConvLSTM"):
                if name in results and "se_pred" in results[name]:
                    ax.plot(steps, results[name]["se_pred"], marker="o", label=name)
            if "MeanFlow" in results and "se_perfect" in results["MeanFlow"]:
                ax.plot(steps, results["MeanFlow"]["se_perfect"], "k--", label="perfect CSI")
        ax.set_xlabel("prediction step"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, fname), dpi=150); plt.close(fig)

    # outage-constrained goodput + empirical outage (risk-aware link adaptation)
    if any("goodput" in results.get(n, {}) for n in ("MeanFlow", "DiU", "ConvLSTM")):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
        for name in ("MeanFlow", "DiU", "ConvLSTM"):
            if name in results and "goodput" in results[name]:
                a1.plot(steps, results[name]["goodput"], marker="o", label=name)
                a2.plot(steps, results[name]["outage"], marker="o", label=name)
        a2.axhline(args.epsilon, color="k", ls="--", lw=1, label=f"target {args.epsilon:g}")
        a1.set_title(f"Goodput @ {args.epsilon:g}-outage"); a1.set_ylabel("bits/s/Hz")
        a2.set_title("Empirical outage"); a2.set_ylabel("outage prob")
        for a in (a1, a2):
            a.set_xlabel("prediction step"); a.grid(True, alpha=0.3); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "outage.png"), dpi=150); plt.close(fig)

    with open(os.path.join(args.out_dir, "uncertainty.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved figures + uncertainty.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
