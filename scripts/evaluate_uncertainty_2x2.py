"""Final 2x2 mu-ablation evaluation: NMSE + reliability + downstream, physical space.

Loads the four generative cells -- {MeanFlow, diffusion} x {mu, no-mu}, all on the
identical TemporalEncoder + UNetGenerator (~2.2M) -- plus the AR ConvLSTM point
reference, evaluates them on ONE shared set of Sionna channels, and reports per model:

    ensemble-mean NMSE, CRPS, coverage/reliability, spread-skill, outage-goodput, SE.

MeanFlow draws K stochastic 1-NFE samples; diffusion draws K stochastic DDIM samples
(--diff-steps, default 3) via the shared-backbone residual/standard path; ConvLSTM is
the deterministic point baseline (NMSE + SE + degenerate reliability, for contrast).

Usage (Alvis):
    python -m scripts.evaluate_uncertainty_2x2 --K 30 --snr 20 --diff-steps 3 \
        --mf-mu runs/mf_6913678/ckpt_best.pt --mf-nomu runs/ar_meanflow_muoff_.../ckpt_best.pt \
        --diff-mu runs/ar_diffusion_muon_.../ckpt_best.pt --diff-nomu runs/ar_diffusion_muoff_.../ckpt_best.pt \
        --convlstm runs/arlstm_6916298/ckpt_best.pt
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
from mf_csi.data.dataset import _normalize, _split_batch, denormalize
from mf_csi.models import TemporalEncoder, UNetGenerator, ARConvLSTM
from mf_csi.diffusion import make_scheduler, corrupt_history
from mf_csi.diffusion_shared import ddim_ar_predict_shared
from mf_csi.inference import autoregressive_predict, ar_convlstm_predict, nmse, nmse_db
from mf_csi.uncertainty import (crps, coverage, spread_skill, ensemble_mean_nmse,
                                spectral_efficiency, outage_operating_curve, goodput_at_outage,
                                outage_global, selected_rates, crps_rate)

LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
# name -> (color, is_point)
STYLE = {"MeanFlow+mu": ("#d62728", False), "MeanFlow-mu": ("#f0997b", False),
         "Diffusion+mu": ("#1f77b4", False), "Diffusion-mu": ("#85b7eb", False),
         "ConvLSTM": ("#2ca02c", True)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mf-mu", type=str, default=None)
    p.add_argument("--mf-nomu", type=str, default=None)
    p.add_argument("--diff-mu", type=str, default=None)
    p.add_argument("--diff-nomu", type=str, default=None)
    p.add_argument("--convlstm", type=str, default=None)
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--diff-steps", type=int, default=3)
    p.add_argument("--ddim-eta", type=float, default=1.0,
                   help="DDIM stochasticity: 1.0=stochastic (default, for UQ), 0.0=deterministic (best NMSE).")
    p.add_argument("--step-noise", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--n-samples", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", type=str, default="runs/uq2x2")
    return p.parse_args()


def raw_batches(cfg, n, bs):
    gen = CDLChannelGenerator(cfg); out = []; rem = n
    while rem > 0:
        b = min(bs, rem); out.append(gen.generate(b)); rem -= b
    return out


def to_batches(raw, cfg):
    out = []
    for c in raw:
        c_n, stats = _normalize(c, "std", None)
        out.append(_split_batch(c_n, cfg, stats))
    return out


def denorm_ens(samples, stats):
    return torch.stack([denormalize(samples[k], stats) for k in range(samples.shape[0])], dim=0)


def load_gen(ckpt, cfg, device):
    ck = torch.load(ckpt, map_location=device)
    enc = TemporalEncoder(cfg.encoder).to(device); gen = UNetGenerator(cfg.generator).to(device)
    enc.load_state_dict(ck["ema_enc"]); gen.load_state_dict(ck["ema_gen"]); enc.eval(); gen.eval()
    return enc, gen


def metrics_for(sample_fn, batches, device, snr, args, is_point=False):
    agg = {}
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = corrupt_history(past, snr, snr)
        s = denorm_ens(sample_fn(hist, future.shape[1]), b["stats"])   # [K,B,Nf,2,Nt,Nc]
        y = denormalize(future, b["stats"])
        agg.setdefault("nmse", []).append(ensemble_mean_nmse(s, y)[0])
        agg.setdefault("se_pred", []).append(spectral_efficiency(s.mean(0), y, snr)[0])
        oc_o, oc_g = outage_operating_curve(s, y, snr)
        agg.setdefault("oc_outage", []).append(oc_o); agg.setdefault("oc_goodput", []).append(oc_g)
        R_i, ct = selected_rates(s, y, snr, args.epsilon)
        agg.setdefault("_R", []).append(R_i); agg.setdefault("_ct", []).append(ct)
        gR, gO, gG = outage_global(s, y, snr, args.epsilon)
        agg.setdefault("_gR", []).append(gR); agg.setdefault("_gO", []).append(gO); agg.setdefault("_gG", []).append(gG)
        for lv in LEVELS:
            agg.setdefault(f"cov{lv}", []).append(coverage(s, y, lv)[1].item())
        agg.setdefault("crps_rate", []).append(crps_rate(s, y, snr)[0])
        if not is_point:
            agg.setdefault("crps", []).append(crps(s, y)[0])
            agg.setdefault("spread", []).append(spread_skill(s, y)[0])
            agg.setdefault("skill", []).append(spread_skill(s, y)[1])
    Rflat = torch.cat(agg.pop("_R")); ctflat = torch.cat(agg.pop("_ct"))
    gR = float(np.mean(agg.pop("_gR"))); gO = float(np.mean(agg.pop("_gO"))); gG = float(np.mean(agg.pop("_gG")))
    out = {}
    for k, v in agg.items():
        out[k] = float(np.mean(v)) if k.startswith("cov") else torch.stack(v).mean(0).tolist()
    if "oc_outage" in out:
        out["goodput_at_eps"] = goodput_at_outage(torch.tensor(out["oc_outage"]),
                                                  torch.tensor(out["oc_goodput"]), args.epsilon)
    # rate diagnostics
    qs = torch.linspace(0.0, 1.0, 101, device=Rflat.device)
    out["R_mean"] = float(Rflat.mean()); out["R_std"] = float(Rflat.std())
    out["R_cdf"] = torch.quantile(Rflat, qs).tolist()
    out["ctrue_cdf"] = torch.quantile(ctflat, qs).tolist()
    out["cdf_p"] = qs.tolist()
    out["global_goodput"] = gG; out["global_outage"] = gO; out["global_R"] = gR
    return out


def main():
    args = parse_args()
    cfg = Config()
    cfg.diu.sampling_steps = args.diff_steps
    cfg.diu.deterministic_init = False; cfg.diu.ddim_eta = args.ddim_eta  # eta=1 stochastic (UQ), eta=0 deterministic (best NMSE)
    # random init keeps an ensemble even at eta=0, so coverage still reports the calibration cost
    cfg.inference.seed_std = cfg.meanflow.source_std
    scheduler = make_scheduler(cfg.diu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    _mode = "deterministic" if args.ddim_eta == 0.0 else "stochastic"
    print(f"shared eval set: {args.n_samples} | K={args.K} | SNR={args.snr} | diff_steps={args.diff_steps} | ddim_eta={args.ddim_eta} ({_mode})")
    raw = raw_batches(cfg.data, args.n_samples, args.batch_size)
    batches = to_batches(raw, cfg.data)

    results = {}

    def add_meanflow(name, ckpt, use_mu):
        if not ckpt: return
        enc, gen = load_gen(ckpt, cfg, device)
        def sfn(hist, Nf):
            return torch.stack([autoregressive_predict(enc, gen, hist, Nf,
                                seed_std=cfg.inference.seed_std, step_noise_std=args.step_noise,
                                num_samples=1, use_mu=use_mu) for _ in range(args.K)], dim=0)
        results[name] = metrics_for(sfn, batches, device, args.snr, args)
        print(f"[{name}] NMSE {nmse_db(torch.tensor(results[name]['nmse']).mean()).item():.2f} dB "
              f"| cov0.9 {results[name]['cov0.9']:.3f} | goodput@{args.epsilon:.0%} {results[name]['goodput_at_eps']:.2f}")

    def add_diffusion(name, ckpt, use_mu):
        if not ckpt: return
        enc, gen = load_gen(ckpt, cfg, device)
        def sfn(hist, Nf):
            return torch.stack([ddim_ar_predict_shared(enc, gen, scheduler, hist, Nf, cfg.diu,
                                use_mu=use_mu) for _ in range(args.K)], dim=0)
        results[name] = metrics_for(sfn, batches, device, args.snr, args)
        print(f"[{name}] NMSE {nmse_db(torch.tensor(results[name]['nmse']).mean()).item():.2f} dB "
              f"| cov0.9 {results[name]['cov0.9']:.3f} | goodput@{args.epsilon:.0%} {results[name]['goodput_at_eps']:.2f}")

    add_meanflow("MeanFlow+mu", args.mf_mu, True)
    add_meanflow("MeanFlow-mu", args.mf_nomu, False)
    add_diffusion("Diffusion+mu", args.diff_mu, True)
    add_diffusion("Diffusion-mu", args.diff_nomu, False)
    if args.convlstm:
        ck = torch.load(args.convlstm, map_location=device)
        cl = ARConvLSTM(cfg.ar_convlstm).to(device); cl.load_state_dict(ck.get("ema", ck.get("model"))); cl.eval()
        def sfn(hist, Nf): return ar_convlstm_predict(cl, hist, Nf).unsqueeze(0)   # K=1 point
        results["ConvLSTM"] = metrics_for(sfn, batches, device, args.snr, args, is_point=True)
        print(f"[ConvLSTM] NMSE {nmse_db(torch.tensor(results['ConvLSTM']['nmse']).mean()).item():.2f} dB (point)")

    steps = list(range(1, cfg.data.num_future + 1))
    # (1) NMSE per step -- ALL models incl ConvLSTM
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, r in results.items():
        c = STYLE[name][0]
        ax.plot(steps, [nmse_db(torch.tensor(v)).item() for v in r["nmse"]], marker="o", ms=3,
                color=c, linestyle=("--" if STYLE[name][1] else "-"), label=name)
    ax.set_xlabel("prediction step"); ax.set_ylabel("NMSE (dB), physical"); ax.set_title("2x2 + ConvLSTM: NMSE")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "nmse.png"), dpi=150); plt.close(fig)
    # (2) reliability
    fig, ax = plt.subplots(figsize=(4.8, 4.4)); ax.plot([0, 1], [0, 1], "k:", lw=1, label="ideal")
    for name, r in results.items():
        if f"cov{LEVELS[0]}" in r:
            ax.plot(LEVELS, [r[f"cov{lv}"] for lv in LEVELS], marker="o", color=STYLE[name][0], label=name)
    ax.set_xlabel("nominal"); ax.set_ylabel("empirical"); ax.set_title("Reliability")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "reliability.png"), dpi=150); plt.close(fig)
    # (3) goodput-vs-outage OPERATING CURVE (bias-robust)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, r in results.items():
        if "oc_outage" in r:
            ax.plot(r["oc_outage"], r["oc_goodput"], marker="o", ms=3, color=STYLE[name][0], label=name)
    ax.axvline(args.epsilon, color="k", ls="--", lw=1, label=f"target outage {args.epsilon:g}")
    ax.set_xlabel("achieved outage"); ax.set_ylabel("goodput (b/s/Hz)")
    ax.set_title("Rate-adaptation operating curve (up-left is better)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "operating_curve.png"), dpi=150); plt.close(fig)

    # (4) rate CDF diagnostic: are some models simply selecting more aggressive rates?
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ct_ref = None
    for name, r in results.items():
        if "R_cdf" in r:
            ax.plot(r["R_cdf"], r["cdf_p"], color=STYLE[name][0], lw=2, label=name)
            ct_ref = r["ctrue_cdf"]; p_ref = r["cdf_p"]
    if ct_ref is not None:
        ax.plot(ct_ref, p_ref, color="black", lw=1.5, ls="--", label="true rate")
    ax.set_xlabel("selected rate $R_i$ (bits/s/Hz)"); ax.set_ylabel("CDF")
    ax.set_title(f"Selected per-coefficient rate at q={args.epsilon:g}")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "rate_cdf.png"), dpi=150); plt.close(fig)

    print("\n=== rate diagnostics (aggression vs variance) & Case-1 global goodput ===")
    for name, r in results.items():
        if "R_mean" in r:
            print(f"  {name:13s} | R_i mean {r['R_mean']:.2f} std {r['R_std']:.2f} "
                  f"| Case2 goodput@eps {r.get('goodput_at_eps', float('nan')):.2f} "
                  f"| Case1 goodput {r['global_goodput']:.2f} | CRPS-rate {float(np.mean(r['crps_rate'])):.3f}")

    with open(os.path.join(args.out_dir, "uncertainty_2x2.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved nmse.png, reliability.png, outage.png, uncertainty_2x2.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
