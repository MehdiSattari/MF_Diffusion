"""Head-to-head NMSE-vs-prediction-step comparison: diffusion DiU vs MeanFlow DiU.

Both models are evaluated on the SAME underlying Sionna channels (generated once,
then normalized in each model's own convention) and rolled out autoregressively
over the Nf-frame horizon. Per-step NMSE (dB) is computed for each model at a
"clean" history and at one or more inference SNRs, overlaid on a single figure.

Why a dedicated script: the training logs only recorded step-1 and step-10, and
the two models live in different normalization spaces (DiU: global min-max fit
once; MeanFlow: per-sample min-max). Sharing one raw eval set and evaluating each
model in its own trained space gives a like-for-like, paper-consistent curve.

Outputs (into --out-dir, default runs/compare/):
    nmse_vs_step_compare.png   overlaid DiU vs MeanFlow, one line per (model, SNR)
    nmse_vs_step_compare.json  the raw per-step NMSE (dB) for replotting

Usage (on Alvis, inside the venv):
    python -m scripts.evaluate_compare \
        --diu-ckpt runs/diu_6793522/ckpt_best.pt \
        --mf-ckpt  runs/mf_6795740/ckpt_best.pt \
        --snrs 20 --n-samples 256
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mf_csi.config import Config
from mf_csi.data.sionna_cdl import CDLChannelGenerator
from mf_csi.data.dataset import _normalize, _split_batch, estimate_global_minmax, denormalize
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.models.diu import DiUEncoder, DiUNet
from mf_csi.diffusion import make_scheduler, ddim_ar_predict, corrupt_history
from mf_csi.inference import autoregressive_predict, mu_only_predict, nmse, nmse_db


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diu-ckpt", type=str, required=True, help="diffusion DiU checkpoint")
    p.add_argument("--mf-ckpt", type=str, required=True, help="MeanFlow DiU checkpoint")
    p.add_argument("--out-dir", type=str, default="runs/compare")
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--snrs", type=str, default="20",
                   help="comma-separated inference SNRs in dB (history corruption)")
    p.add_argument("--clean", action="store_true",
                   help="also evaluate with a clean (uncorrupted) history")
    p.add_argument("--mf-mean-samples", type=int, default=None,
                   help="1-NFE draws averaged per MeanFlow frame (default: config)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Shared raw eval set  (generate the channels ONCE, normalize per model)
# --------------------------------------------------------------------------- #
def raw_eval_batches(dcfg, n_samples, batch_size):
    """Generate a fixed list of RAW CSI batches [B, T, 2, Nt, Nc] (pre-normalization)."""
    gen = CDLChannelGenerator(dcfg)
    batches, remaining = [], n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        batches.append(gen.generate(b))
        remaining -= b
    return batches


def to_model_batches(raw_batches, dcfg, mode, global_ab):
    """Normalize each raw batch in the given convention and split past/future."""
    out = []
    for csi in raw_batches:
        csi_n, stats = _normalize(csi, mode, global_ab)
        out.append(_split_batch(csi_n, dcfg, stats))
    return out


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_diu(ckpt_path, cfg, device, weights):
    ck = torch.load(ckpt_path, map_location=device)
    enc = DiUEncoder(cfg.diu, in_channels=2).to(device)
    unet = DiUNet(cfg.diu, data_channels=2, image_size=cfg.data.num_subcarriers_used).to(device)
    ek, uk = ("ema_enc", "ema_unet") if weights == "ema" else ("enc", "unet")
    enc.load_state_dict(ck[ek]); unet.load_state_dict(ck[uk])
    enc.eval(); unet.eval()
    print(f"[DiU]      {weights} weights | step {ck.get('step')} | "
          f"best {ck.get('best_nmse_db')} | global_ab {ck.get('global_ab')}")
    return enc, unet, ck.get("global_ab")


def load_meanflow(ckpt_path, cfg, device, weights):
    ck = torch.load(ckpt_path, map_location=device)
    enc = TemporalEncoder(cfg.encoder).to(device)
    gen = UNetGenerator(cfg.generator).to(device)
    ek, gk = ("ema_enc", "ema_gen") if weights == "ema" else ("enc", "gen")
    enc.load_state_dict(ck[ek]); gen.load_state_dict(ck[gk])
    enc.eval(); gen.eval()
    sigma = ck.get("source_std")
    print(f"[MeanFlow] {weights} weights | step {ck.get('step')} | "
          f"best {ck.get('best_nmse_db')} | source_std {sigma}")
    return enc, gen, sigma


# --------------------------------------------------------------------------- #
# Per-step NMSE.  raw=True computes NMSE in the DENORMALIZED physical-channel
# space (each model's prediction + target mapped back through its own stats), so
# the two models -- trained in DIFFERENT normalization spaces -- become directly
# comparable. raw=False keeps each model's own normalized-space NMSE.
# --------------------------------------------------------------------------- #
def _nmse(pred, future, stats, raw):
    if raw:
        pred = denormalize(pred, stats)
        future = denormalize(future, stats)
    ps, _ = nmse(pred, future)
    return ps


@torch.no_grad()
def diu_per_step(enc, unet, scheduler, batches, cfg, device, snr, raw=False):
    ps_sum = None
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = ddim_ar_predict(enc, unet, scheduler, hist, future.shape[1], cfg.diu)
        ps = _nmse(pred, future, b["stats"], raw)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    return ps_sum / len(batches)


@torch.no_grad()
def meanflow_per_step(enc, gen, batches, cfg, device, snr, raw=False):
    ps_sum = None
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = autoregressive_predict(
            enc, gen, hist, future.shape[1],
            seed_std=cfg.inference.seed_std,
            step_noise_std=cfg.inference.step_noise_std,
            num_samples=cfg.inference.mean_samples)
        ps = _nmse(pred, future, b["stats"], raw)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    return ps_sum / len(batches)


@torch.no_grad()
def meanflow_mu_per_step(enc, batches, cfg, device, snr, raw=False):
    """Ablation: NMSE using ONLY the encoder's point estimate mu (no flow) -- the
    regression baseline embedded in the informative-prior model."""
    ps_sum = None
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = mu_only_predict(enc, hist, future.shape[1])
        ps = _nmse(pred, future, b["stats"], raw)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    return ps_sum / len(batches)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_compare(curves, path, space="physical"):
    """curves: list of (label, per_step_linear_tensor, color, linestyle)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for label, ps, color, ls in curves:
        ax.plot(range(1, len(ps) + 1), [nmse_db(v).item() for v in ps],
                marker="o", ms=4, color=color, linestyle=ls, label=label)
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("prediction step")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"NMSE vs prediction step — DiU vs MeanFlow ({space} space)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config()
    if args.mf_mean_samples is not None:
        cfg.inference.mean_samples = args.mf_mean_samples
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # --- load both models ---
    enc_d, unet_d, global_ab = load_diu(args.diu_ckpt, cfg, device, args.weights)
    enc_m, gen_m, sigma = load_meanflow(args.mf_ckpt, cfg, device, args.weights)
    cfg.meanflow.source_std = sigma if sigma is not None else cfg.meanflow.source_std
    cfg.inference.seed_std = cfg.meanflow.source_std
    scheduler = make_scheduler(cfg.diu)

    # The DiU checkpoint may predate global_ab being saved (None). Reproduce the
    # training-time behaviour: estimate the global min-max once on the same data
    # config. The CDL CSI distribution is stationary, so a 4000-sample estimate
    # matches the value the model was trained under to within rounding.
    if global_ab is None:
        print("DiU checkpoint has no global_ab; estimating global min-max ...")
        global_ab = estimate_global_minmax(cfg.data, num_samples=4000, batch_size=256)
        print(f"estimated global (a, b) = ({global_ab[0]:.4f}, {global_ab[1]:.4f})")

    # --- one shared raw eval set (identical channels for both models) ---
    print(f"generating shared eval set: {args.n_samples} samples ...")
    raw = raw_eval_batches(cfg.data, args.n_samples, args.batch_size)
    diu_batches = to_model_batches(raw, cfg.data, "global_minmax11", global_ab)
    mf_batches = to_model_batches(raw, cfg.data, "std", None)  # informative-prior MeanFlow

    snrs = [float(s) for s in args.snrs.split(",") if s.strip()]
    conditions = ([(None, "clean")] if args.clean else []) + [(s, f"{s:g} dB") for s in snrs]

    palette = {"DiU": "#1f77b4", "MeanFlow": "#d62728", "MeanFlow-mu": "#7f7f7f"}
    styles = ["-", "--", "-.", ":"]
    # Physical (denormalized) space is the FAIR comparison -- both models mapped
    # back to the raw channel. Normalized-space numbers are kept for continuity.
    dump = {"steps": None, "note": "physical = denormalized (fair, cross-model); "
            "normalized = each model's own training space", "series": []}
    curves_phys, curves_norm = [], []

    print("\n=== per-step NMSE (dB) ===  [physical space = fair comparison]")
    for k, (snr, tag) in enumerate(conditions):
        ls = styles[k % len(styles)]
        for space, raw, curves in (("physical", True, curves_phys),
                                    ("normalized", False, curves_norm)):
            ps_d = diu_per_step(enc_d, unet_d, scheduler, diu_batches, cfg, device, snr, raw=raw)
            ps_m = meanflow_per_step(enc_m, gen_m, mf_batches, cfg, device, snr, raw=raw)
            ps_mu = meanflow_mu_per_step(enc_m, mf_batches, cfg, device, snr, raw=raw)
            dump["steps"] = list(range(1, len(ps_d) + 1))
            for name, ps in [("DiU", ps_d), ("MeanFlow", ps_m), ("MeanFlow-mu", ps_mu)]:
                db = [round(nmse_db(v).item(), 3) for v in ps]
                avg_db = round(nmse_db(ps.mean()).item(), 3)
                curves.append((f"{name} ({tag})", ps, palette[name], ls))
                dump["series"].append({"model": name, "space": space, "condition": tag,
                                       "nmse_db": db, "avg_nmse_db": avg_db})
                if space == "physical":
                    print(f"  {name:12s} ({tag:>6s}) | step1 {db[0]:7.2f} | "
                          f"step{len(db)} {db[-1]:7.2f} | avg {avg_db:7.2f}")

    png = os.path.join(args.out_dir, "nmse_vs_step_compare.png")            # physical (fair)
    png_norm = os.path.join(args.out_dir, "nmse_vs_step_compare_norm.png")  # normalized
    js = os.path.join(args.out_dir, "nmse_vs_step_compare.json")
    plot_compare(curves_phys, png, space="physical")
    plot_compare(curves_norm, png_norm, space="normalized")
    with open(js, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nsaved: {png}\n       {png_norm}\n       {js}")


if __name__ == "__main__":
    main()
