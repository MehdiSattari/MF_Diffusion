"""Evaluate a trained MeanFlow DiU checkpoint.

Produces:
  * NMSE vs prediction step (per inference SNR)               -> nmse_vs_step.png
  * NMSE vs inference SNR (1st step / last step / average)    -> nmse_vs_snr.png
  * Ground-truth vs predicted CSI over the horizon (|H|)      -> csi_pred_vs_gt.png
    (rows alternate GT / prediction per sample, cols = steps -- the paper's Fig 6/7)
  * a text summary of per-step NMSE (dB).

Inference-SNR note: to test robustness we corrupt the observed history with the
same augmentation used in training (X~ = sqrt(rho) X + N). Only the observed
frames are corrupted; autoregressively predicted frames re-enter clean.

Usage (on Alvis):
    python -m scripts.evaluate --ckpt runs/mf_<jobid>/ckpt_best.pt --snrs 0,10,20
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mf_csi.config import Config
from mf_csi.data import make_fixed_eval_set
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.inference import autoregressive_predict, nmse, nmse_db
from mf_csi.meanflow import augment_history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--snrs", type=str, default="0,10,20",
                   help="comma-separated inference SNRs in dB; empty for clean-only")
    p.add_argument("--n-show", type=int, default=3, help="samples in the GT-vs-pred figure")
    return p.parse_args()


def load_models(ckpt_path, cfg, device, weights):
    ck = torch.load(ckpt_path, map_location=device)
    enc = TemporalEncoder(cfg.encoder).to(device)
    gen = UNetGenerator(cfg.generator).to(device)
    ekey, gkey = ("ema_enc", "ema_gen") if weights == "ema" else ("enc", "gen")
    enc.load_state_dict(ck[ekey]); gen.load_state_dict(ck[gkey])
    enc.eval(); gen.eval()
    print(f"loaded {weights} weights from {ckpt_path} (step {ck.get('step')}, "
          f"best {ck.get('best_nmse_db')})")
    return enc, gen


def csi_magnitude(x):
    return torch.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)   # [B, Nf, Nt, Nc]


def corrupt_history(past, snr_db, cfg):
    mf = replace(cfg.meanflow, noise_aug=True, snr_db_min=snr_db, snr_db_max=snr_db)
    return augment_history(past, mf)


@torch.no_grad()
def eval_nmse(enc, gen, batches, cfg, device, snr_db=None):
    ps_sum = None
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr_db is None else corrupt_history(past, snr_db, cfg)
        pred = autoregressive_predict(enc, gen, hist, future.shape[1],
                                      seed_std=cfg.inference.seed_std,
                                      step_noise_std=cfg.inference.step_noise_std)
        ps, _ = nmse(pred, future)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    ps = ps_sum / len(batches)
    return ps, ps.mean()


def plot_nmse_vs_step(curves, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label, ps in curves:
        ax.plot(range(1, len(ps) + 1), [nmse_db(v).item() for v in ps],
                marker="o", ms=3, label=label)
    ax.set_xlabel("prediction step"); ax.set_ylabel("NMSE (dB)")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_title("NMSE vs prediction step")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_nmse_vs_snr(snrs, avg, s1, sN, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(snrs, [nmse_db(v).item() for v in s1], marker="o", label="1st step")
    ax.plot(snrs, [nmse_db(v).item() for v in sN], marker="s", label="last step")
    ax.plot(snrs, [nmse_db(v).item() for v in avg], marker="^", label="average")
    ax.set_xlabel("inference SNR (dB)"); ax.set_ylabel("NMSE (dB)")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_title("NMSE vs inference SNR")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_pred_vs_gt(pred, future, path, n_show=3):
    magp = csi_magnitude(pred).cpu().numpy()
    magt = csi_magnitude(future).cpu().numpy()
    Nf = magp.shape[1]
    n_show = min(n_show, magp.shape[0])
    rows = 2 * n_show
    fig, axes = plt.subplots(rows, Nf, figsize=(1.3 * Nf, 1.4 * rows))
    axes = np.atleast_2d(axes)
    for i in range(n_show):
        vmin = min(magt[i].min(), magp[i].min())
        vmax = max(magt[i].max(), magp[i].max())
        for j in range(Nf):
            axes[2 * i, j].imshow(magt[i, j], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            axes[2 * i + 1, j].imshow(magp[i, j], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            for r in (2 * i, 2 * i + 1):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if i == 0:
                axes[0, j].set_title(f"step {j + 1}", fontsize=8)
        axes[2 * i, 0].set_ylabel(f"GT s{i}", fontsize=8)
        axes[2 * i + 1, 0].set_ylabel(f"Pred s{i}", fontsize=8)
    fig.suptitle("Ground truth vs MeanFlow prediction over the horizon  (|H|, ant x sc)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    args = parse_args()
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out_dir or os.path.join(os.path.dirname(args.ckpt), "eval")
    os.makedirs(out, exist_ok=True)

    enc, gen = load_models(args.ckpt, cfg, device, args.weights)
    batches = make_fixed_eval_set(cfg.data, args.n_samples, args.batch_size)

    # NMSE vs step, clean history.
    ps_clean, avg_clean = eval_nmse(enc, gen, batches, cfg, device, snr_db=None)
    print("\nNMSE per step (clean history):")
    for n, v in enumerate(ps_clean, 1):
        print(f"  step {n:2d}: {nmse_db(v).item():6.2f} dB")
    print(f"average: {nmse_db(avg_clean).item():.2f} dB")

    curves = [("clean", ps_clean)]
    snrs = [float(s) for s in args.snrs.split(",") if s.strip() != ""]
    if snrs:
        avg, s1, sN = [], [], []
        for snr in snrs:
            ps, av = eval_nmse(enc, gen, batches, cfg, device, snr_db=snr)
            curves.append((f"SNR {snr:g} dB", ps))
            avg.append(av); s1.append(ps[0]); sN.append(ps[-1])
            print(f"SNR {snr:g} dB | avg NMSE {nmse_db(av).item():.2f} dB")
        plot_nmse_vs_snr(snrs, avg, s1, sN, os.path.join(out, "nmse_vs_snr.png"))

    plot_nmse_vs_step(curves, os.path.join(out, "nmse_vs_step.png"))

    # Ground-truth vs prediction figure (clean, first eval batch).
    b = batches[0]
    past, future = b["past"].to(device), b["future"].to(device)
    pred = autoregressive_predict(enc, gen, past, future.shape[1],
                                  seed_std=cfg.inference.seed_std,
                                  step_noise_std=cfg.inference.step_noise_std)
    plot_pred_vs_gt(pred, future, os.path.join(out, "csi_pred_vs_gt.png"), n_show=args.n_show)

    print(f"\nsaved to {out}/: nmse_vs_step.png"
          + (", nmse_vs_snr.png" if snrs else "")
          + ", csi_pred_vs_gt.png")


if __name__ == "__main__":
    main()
