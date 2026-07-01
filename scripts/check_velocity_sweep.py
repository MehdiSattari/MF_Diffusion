"""Temporal decorrelation of the Sionna CDL channel at FIXED velocities.

Fixes the CDL profile (CDL-C, 100 ns delay spread) and varies only the user
velocity {5, 30, 60, 120} km/h, so the difference in decorrelation is purely a
Doppler effect. Higher velocity -> faster temporal decorrelation.

Saves (to viz/):
  * csi_velocity_decorrelation.png : |temporal correlation| vs lag, one curve per speed.
  * csi_velocity_grid.png          : |H| over time, one row per speed (sample 0).
  * csi_v5.gif, csi_v120.gif       : slowest vs fastest, animated.

Run on Alvis:  python -m scripts.check_velocity_sweep
"""

from __future__ import annotations

import os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from mf_csi.config import Config
from mf_csi.data.sionna_cdl import CDLChannelGenerator

VELOCITIES = [5, 30, 60, 120]      # km/h
BATCH = 16
FIXED_CDL = "C"
FIXED_DELAY_SPREAD_NS = 100.0


def complex_magnitude(csi: np.ndarray) -> np.ndarray:
    return np.sqrt(csi[:, :, 0] ** 2 + csi[:, :, 1] ** 2)


def temporal_corr(csi: np.ndarray) -> np.ndarray:
    H = (csi[:, :, 0] + 1j * csi[:, :, 1])
    B, T = H.shape[:2]
    H = H.reshape(B, T, -1)
    corr = np.zeros(T)
    for lag in range(T):
        vals = [np.abs(np.vdot(H[b, 0], H[b, lag])) /
                (np.linalg.norm(H[b, 0]) * np.linalg.norm(H[b, lag]) + 1e-12)
                for b in range(B)]
        corr[lag] = np.mean(vals)
    return corr


def velocity_grid(items, path, times):
    """Rows = velocities, cols = the given (consecutive) frames, sample 0."""
    n = len(items)
    times = list(times)
    fig, axes = plt.subplots(n, len(times), figsize=(1.35 * len(times), 1.5 * n))
    axes = np.atleast_2d(axes)
    for i, (label, csi) in enumerate(items):
        mag = complex_magnitude(csi)[0]
        vmin, vmax = mag[times].min(), mag[times].max()
        for j, t in enumerate(times):
            ax = axes[i, j]
            ax.imshow(mag[t], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t={t}", fontsize=8)
        axes[i, 0].set_ylabel(label, fontsize=9)
    fig.suptitle("CDL |H| over 10 consecutive frames at fixed velocity (rows: speed)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_gif(csi, path, sample=0, fps=6):
    mag = complex_magnitude(csi)[sample]
    T = mag.shape[0]
    vmin, vmax = mag.min(), mag.max()
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    im = ax.imshow(mag[0], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xlabel("subcarrier"); ax.set_ylabel("antenna")
    title = ax.set_title("t = 0")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    def update(t):
        im.set_data(mag[t]); title.set_text(f"t = {t}")
        return [im, title]

    animation.FuncAnimation(fig, update, frames=T, blit=False).save(
        path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main():
    cfg = Config()
    out = "viz"
    os.makedirs(out, exist_ok=True)

    results, grids = [], []
    for v in VELOCITIES:
        dcfg = replace(cfg.data,
                       min_speed_kmh=float(v), max_speed_kmh=float(v),
                       cdl_models=(FIXED_CDL,),
                       min_delay_spread_ns=FIXED_DELAY_SPREAD_NS,
                       max_delay_spread_ns=FIXED_DELAY_SPREAD_NS)
        csi = CDLChannelGenerator(dcfg).generate(BATCH)
        corr = temporal_corr(csi)
        results.append((v, corr))
        grids.append((f"{v} km/h", csi))
        print(f"v={v:3d} km/h | corr[last]={corr[-1]:.3f}")

    # Combined decorrelation figure.
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for v, corr in results:
        ax.plot(np.arange(len(corr)), corr, marker="o", ms=3, label=f"{v} km/h")
    ax.set_xlabel("time lag (frames)")
    ax.set_ylabel("|temporal correlation|")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(title="velocity")
    ax.set_title(f"CSI temporal decorrelation vs velocity (28 GHz, CDL-{FIXED_CDL})")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "csi_velocity_decorrelation.png"), dpi=140)
    plt.close(fig)

    future_times = range(cfg.data.num_past, cfg.data.num_past + cfg.data.num_future)
    velocity_grid(grids, os.path.join(out, "csi_velocity_grid.png"), future_times)
    save_gif(grids[0][1], os.path.join(out, f"csi_v{VELOCITIES[0]}.gif"))
    save_gif(grids[-1][1], os.path.join(out, f"csi_v{VELOCITIES[-1]}.gif"))

    finals = [corr[-1] for _, corr in results]
    assert finals[0] == max(finals), f"slowest should stay most correlated: {finals}"
    assert finals[0] - finals[-1] > 0.1, f"expected clear speed separation: {finals}"
    print("OK: higher velocity -> faster decorrelation.")
    print(f"saved to {out}/: csi_velocity_decorrelation.png, csi_velocity_grid.png, "
          f"csi_v{VELOCITIES[0]}.gif, csi_v{VELOCITIES[-1]}.gif")


if __name__ == "__main__":
    main()
