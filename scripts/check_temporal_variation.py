"""Verify that the Sionna CDL data is genuinely time-varying, and save visuals.

Checks (quantitative):
  * temporal std of |H| across frames is non-negligible (not static / not repeated),
  * consecutive-frame relative change,
  * temporal correlation |<H_0, H_t>| decays with lag t (Doppler decorrelation).

Saves (to viz/):
  * csi_temporal_grid.png : |H| heatmaps (antenna x subcarrier) across time, a few
                            samples per row -- the paper's Fig 6/7 style.
  * csi_temporal_corr.png : temporal correlation vs lag.
  * csi_sample.gif        : one sample's |H| animated over the time axis.

Run on Alvis:  python -m scripts.check_temporal_variation
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from mf_csi.config import Config
from mf_csi.data.sionna_cdl import CDLChannelGenerator


def complex_magnitude(csi: np.ndarray) -> np.ndarray:
    """csi [B, T, 2, Nt, Nc] -> |H| [B, T, Nt, Nc]."""
    return np.sqrt(csi[:, :, 0] ** 2 + csi[:, :, 1] ** 2)


def temporal_stats(csi: np.ndarray) -> dict:
    mag = complex_magnitude(csi)                         # [B, T, Nt, Nc]
    B, T = mag.shape[:2]
    tstd = float(mag.std(axis=1).mean())                 # avg over-time std
    tmean = float(mag.mean())
    rel_diff = float(np.abs(mag[:, 1:] - mag[:, :-1]).mean() / (tmean + 1e-12))

    H = (csi[:, :, 0] + 1j * csi[:, :, 1]).reshape(B, T, -1)   # complex, spatial-flattened
    corr = np.zeros(T)
    for lag in range(T):
        vals = []
        for b in range(B):
            a0, al = H[b, 0], H[b, lag]
            den = np.linalg.norm(a0) * np.linalg.norm(al) + 1e-12
            vals.append(np.abs(np.vdot(a0, al)) / den)
        corr[lag] = np.mean(vals)
    return {"tstd": tstd, "tmean": tmean, "rel_diff": rel_diff, "corr": corr}


def save_grid(csi, path, times, n_samples=3):
    """Grid of |H| heatmaps: rows = samples, cols = the given (consecutive) frames."""
    mag = complex_magnitude(csi)
    B = mag.shape[0]
    n_samples = min(n_samples, B)
    times = list(times)
    fig, axes = plt.subplots(n_samples, len(times),
                             figsize=(1.35 * len(times), 1.5 * n_samples))
    axes = np.atleast_2d(axes)
    for i in range(n_samples):
        vmin, vmax = mag[i, times].min(), mag[i, times].max()
        for j, t in enumerate(times):
            ax = axes[i, j]
            ax.imshow(mag[i, t], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t={t}", fontsize=8)
        axes[i, 0].set_ylabel(f"sample {i}\n(ant x sc)", fontsize=8)
    fig.suptitle("Sionna CDL |H| over the 10 prediction-horizon frames "
                 "(rows: samples, cols: consecutive time steps)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_corr_plot(corr, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(np.arange(len(corr)), corr, marker="o", ms=3)
    ax.set_xlabel("time lag (frames)")
    ax.set_ylabel("|temporal correlation|")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title("CSI temporal decorrelation (Doppler)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_gif(csi, path, sample=0, fps=6):
    mag = complex_magnitude(csi)[sample]                 # [T, Nt, Nc]
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

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main():
    cfg = Config()
    out = "viz"
    os.makedirs(out, exist_ok=True)

    gen = CDLChannelGenerator(cfg.data)
    csi = gen.generate(8)                                 # [8, T, 2, Nt, Nc], raw (unnormalized)
    print("generated CSI:", csi.shape,
          f"(T={csi.shape[1]} frames, {cfg.data.num_bs_ant} ant x {cfg.data.num_subcarriers_used} sc)")

    st = temporal_stats(csi)
    print(f"temporal std of |H|      : {st['tstd']:.4e}")
    print(f"consecutive rel. change  : {st['rel_diff']:.3f}")
    print("temporal correlation vs lag:")
    print("  " + np.array2string(st["corr"], precision=2, floatmode="fixed", max_line_width=120))

    assert st["tstd"] > 1e-6, "CSI does NOT vary over time (static frames?!)"
    assert st["corr"][0] > 0.99, "lag-0 self-correlation should be ~1"
    assert st["corr"][-1] < st["corr"][1], "no temporal decorrelation over the horizon"
    print("OK: CSI is time-varying and decorrelates with lag (genuine Doppler evolution).")

    # Show the 10 consecutive prediction-horizon frames (t = Np .. Np+Nf-1),
    # matching the paper's Fig 6/7 layout.
    future_times = range(cfg.data.num_past, cfg.data.num_past + cfg.data.num_future)
    save_grid(csi, os.path.join(out, "csi_temporal_grid.png"), future_times)
    save_corr_plot(st["corr"], os.path.join(out, "csi_temporal_corr.png"))
    save_gif(csi, os.path.join(out, "csi_sample.gif"))
    print(f"saved: {out}/csi_temporal_grid.png, {out}/csi_temporal_corr.png, {out}/csi_sample.gif")


if __name__ == "__main__":
    main()
