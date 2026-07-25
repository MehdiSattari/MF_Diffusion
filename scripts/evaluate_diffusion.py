"""Evaluate a trained diffusion-DiU checkpoint (paper-style figures).

Produces, into <ckpt_dir>/eval/:
  * nmse_vs_step.png      : NMSE vs prediction step, one curve per inference SNR.
  * nmse_vs_snr.png       : 1st-step / last-step / average NMSE vs SNR.
  * nmse_vs_velocity.png  : NMSE vs step at fixed velocities {30,60,120} km/h (SNR 20 dB)
                            -- directly comparable to the paper's Fig. 4.
  * csi_pred_vs_gt.png    : ground truth vs prediction over the horizon.
  * a printed per-step / per-condition NMSE table.

Usage:
    python -m scripts.evaluate_diffusion --ckpt runs/diu_<jobid>/ckpt_best.pt
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
from mf_csi.models.diu import DiUEncoder, DiUNet
from mf_csi.diffusion import make_scheduler, ddim_ar_predict, corrupt_history
from mf_csi.inference import nmse, nmse_db


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--snrs", type=str, default="0,10,20")
    p.add_argument("--velocities", type=str, default="30,60,120")
    p.add_argument("--vel-snr", type=float, default=20.0, help="SNR for the per-velocity sweep")
    p.add_argument("--n-show", type=int, default=3)
    return p.parse_args()


def load_models(ckpt_path, cfg, device, weights):
    ck = torch.load(ckpt_path, map_location=device)
    enc = DiUEncoder(cfg.diu, in_channels=2).to(device)
    unet = DiUNet(cfg.diu, data_channels=2, image_size=cfg.data.num_subcarriers_used).to(device)
    ek, uk = ("ema_enc", "ema_unet") if weights == "ema" else ("enc", "unet")
    enc.load_state_dict(ck[ek]); unet.load_state_dict(ck[uk])
    enc.eval(); unet.eval()
    global_ab = ck.get("global_ab")
    print(f"loaded {weights} weights from {ckpt_path} (step {ck.get('step')}, "
          f"best {ck.get('best_nmse_db')}, global_ab {global_ab})")
    return enc, unet, global_ab


@torch.no_grad()
def eval_nmse(enc, unet, scheduler, batches, cfg, device, snr=None):
    ps_sum = None
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = ddim_ar_predict(enc, unet, scheduler, hist, future.shape[1], cfg.diu)
        ps, _ = nmse(pred, future)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    ps = ps_sum / len(batches)
    return ps, ps.mean()


def fixed_velocity_batches(cfg, v, n_samples, bs, global_ab):
    dcfg = replace(cfg.data, param_sampling="uniform",
                   min_speed_kmh=float(v), max_speed_kmh=float(v),
                   normalization=cfg.data.normalization)
    return make_fixed_eval_set(dcfg, n_samples, bs, global_ab=global_ab)


def _curve_plot(curves, path, title, xlabel="prediction step"):
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for label, ps in curves:
        ax.plot(range(1, len(ps) + 1), [nmse_db(v).item() for v in ps],
                marker="o", ms=3, label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel("NMSE (dB)")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_nmse_vs_snr(snrs, avg, s1, sN, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.plot(snrs, [nmse_db(v).item() for v in s1], marker="o", label="1st step")
    ax.plot(snrs, [nmse_db(v).item() for v in sN], marker="s", label="last step")
    ax.plot(snrs, [nmse_db(v).item() for v in avg], marker="^", label="average")
    ax.set_xlabel("inference SNR (dB)"); ax.set_ylabel("NMSE (dB)")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_title("NMSE vs inference SNR")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_pred_vs_gt(pred, future, path, n_show=3):
    magp = torch.sqrt(pred[:, :, 0] ** 2 + pred[:, :, 1] ** 2).cpu().numpy()
    magt = torch.sqrt(future[:, :, 0] ** 2 + future[:, :, 1] ** 2).cpu().numpy()
    Nf = magp.shape[1]; n_show = min(n_show, magp.shape[0]); rows = 2 * n_show
    fig, axes = plt.subplots(rows, Nf, figsize=(1.3 * Nf, 1.4 * rows))
    axes = np.atleast_2d(axes)
    for i in range(n_show):
        vmin = min(magt[i].min(), magp[i].min()); vmax = max(magt[i].max(), magp[i].max())
        for j in range(Nf):
            axes[2 * i, j].imshow(magt[i, j], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            axes[2 * i + 1, j].imshow(magp[i, j], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            for r in (2 * i, 2 * i + 1):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if i == 0:
                axes[0, j].set_title(f"step {j + 1}", fontsize=8)
        axes[2 * i, 0].set_ylabel(f"GT s{i}", fontsize=8)
        axes[2 * i + 1, 0].set_ylabel(f"Pred s{i}", fontsize=8)
    fig.suptitle("Ground truth vs diffusion-DiU prediction (|H|)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    args = parse_args()
    cfg = Config()
    cfg.data.normalization = "global_minmax11"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out_dir or os.path.join(os.path.dirname(args.ckpt), "eval")
    os.makedirs(out, exist_ok=True)

    enc, unet, global_ab = load_models(args.ckpt, cfg, device, args.weights)
    scheduler = make_scheduler(cfg.diu)
    batches = make_fixed_eval_set(cfg.data, args.n_samples, args.batch_size, global_ab=global_ab)

    # --- NMSE vs step, per SNR (mixture data) ---
    snrs = [float(s) for s in args.snrs.split(",") if s.strip()]
    curves, avg, s1, sN = [], [], [], []
    print("\n[NMSE vs step | mixture data]")
    for snr in snrs:
        ps, ov = eval_nmse(enc, unet, scheduler, batches, cfg, device, snr=snr)
        curves.append((f"SNR {snr:g} dB", ps)); avg.append(ov); s1.append(ps[0]); sN.append(ps[-1])
        print(f"  SNR {snr:g} dB | avg {nmse_db(ov).item():6.2f} | "
              f"step1 {nmse_db(ps[0]).item():6.2f} | step10 {nmse_db(ps[-1]).item():6.2f}")
    _curve_plot(curves, os.path.join(out, "nmse_vs_step.png"),
                "NMSE vs prediction step (per SNR)")
    plot_nmse_vs_snr(snrs, avg, s1, sN, os.path.join(out, "nmse_vs_snr.png"))

    # --- NMSE vs step, per fixed velocity (SNR = vel-snr) : paper Fig 4 ---
    vels = [float(v) for v in args.velocities.split(",") if v.strip()]
    vcurves = []
    print(f"\n[NMSE vs step | fixed velocity, SNR {args.vel_snr:g} dB]")
    for v in vels:
        vb = fixed_velocity_batches(cfg, v, args.n_samples, args.batch_size, global_ab)
        ps, ov = eval_nmse(enc, unet, scheduler, vb, cfg, device, snr=args.vel_snr)
        vcurves.append((f"{v:g} km/h", ps))
        print(f"  {v:g} km/h | avg {nmse_db(ov).item():6.2f} | "
              f"step1 {nmse_db(ps[0]).item():6.2f} | step10 {nmse_db(ps[-1]).item():6.2f}")
    _curve_plot(vcurves, os.path.join(out, "nmse_vs_velocity.png"),
                f"NMSE vs step at fixed velocity (SNR {args.vel_snr:g} dB)")

    # --- GT vs prediction figure (SNR 20 dB history) ---
    b = batches[0]
    past, future = b["past"].to(device), b["future"].to(device)
    hist = corrupt_history(past, 20.0, 20.0)
    pred = ddim_ar_predict(enc, unet, scheduler, hist, future.shape[1], cfg.diu)
    plot_pred_vs_gt(pred, future, os.path.join(out, "csi_pred_vs_gt.png"), n_show=args.n_show)

    print(f"\nsaved to {out}/: nmse_vs_step.png, nmse_vs_snr.png, "
          f"nmse_vs_velocity.png, csi_pred_vs_gt.png")


if __name__ == "__main__":
    main()
