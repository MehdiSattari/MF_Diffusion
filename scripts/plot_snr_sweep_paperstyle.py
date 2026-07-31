"""Paper-style figures for the wide-posterior SNR sweep (2x2 mu-ablation).

Reads figures/deterministic_ddim/sweep_data.json (produced from the paired SNR x eta
eval sweep) and renders, in the format of the paper's Fig. 2 / Fig. 3:

  fig2_nmse_vs_step_multiSNR.{png,pdf} : per-step NMSE panels at 0/10/20 dB (stochastic)
  fig3_nmse_vs_snr.{png,pdf}           : NMSE vs SNR (first step / last step / average)
  crps_rate_vs_snr.{png,pdf}           : CRPS of achievable rate vs SNR  (the downstream win)
  coverage_vs_snr.{png,pdf}            : coverage@90 vs SNR (calibration; nominal 0.9)

NOT added to the paper -- saved under figures/deterministic_ddim/ for review.
"""
import os, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "figures", "deterministic_ddim")
DATA = json.load(open(os.path.join(OUT, "sweep_data.json")))

MODELS = ["Diffusion+mu", "MeanFlow+mu", "MeanFlow-mu", "Diffusion-mu", "ConvLSTM"]
LABEL = {"MeanFlow+mu": "MeanFlow ($+\\mu$)", "MeanFlow-mu": "MeanFlow ($-\\mu$)",
         "Diffusion+mu": "Diffusion ($+\\mu$)", "Diffusion-mu": "Diffusion ($-\\mu$)",
         "ConvLSTM": "ConvLSTM"}
STYLE = {"MeanFlow+mu": ("#d62728", "o", "-"), "MeanFlow-mu": ("#f0997b", "o", "--"),
         "Diffusion+mu": ("#1f77b4", "s", "-"), "Diffusion-mu": ("#85b7eb", "s", "--"),
         "ConvLSTM": ("#2ca02c", "^", ":")}
SNRS = [0, 5, 10, 20]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "font.family": "serif"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2_multiSNR():
    panels = [0, 10, 20]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    steps = list(range(1, 11))
    for ax, snr in zip(axes, panels):
        rec = DATA["stoch"][str(snr)]
        for m in MODELS:
            if m not in rec:
                continue
            c, mk, ls = STYLE[m]
            ax.plot(steps, rec[m]["nmse_db"], color=c, marker=mk, linestyle=ls,
                    ms=4, lw=1.6, label=LABEL[m])
        ax.set_title(f"SNR = {snr} dB")
        ax.set_xlabel("Prediction step"); ax.set_xlim(1, 10); ax.set_xticks(steps)
    axes[0].set_ylabel("NMSE (dB)")
    axes[-1].legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig2_nmse_vs_step_multiSNR")


def fig3_nmse_vs_snr():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for m in ["Diffusion+mu", "MeanFlow+mu", "MeanFlow-mu", "Diffusion-mu"]:
        c, mk, _ = STYLE[m]
        avg = [sum(DATA["stoch"][str(s)][m]["nmse_db"]) / 10 for s in SNRS]
        first = [DATA["stoch"][str(s)][m]["nmse_db"][0] for s in SNRS]
        last = [DATA["stoch"][str(s)][m]["nmse_db"][-1] for s in SNRS]
        ax.plot(SNRS, avg, color=c, marker=mk, lw=1.8, label=f"{LABEL[m]} (avg)")
        ax.plot(SNRS, first, color=c, marker=mk, lw=1.0, ls=":", alpha=0.6)
        ax.plot(SNRS, last, color=c, marker=mk, lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("Inference SNR (dB)"); ax.set_ylabel("NMSE (dB)")
    ax.set_xticks(SNRS)
    ax.set_title("NMSE vs SNR  (solid=avg, dotted=step 1, dashed=step 10)")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig3_nmse_vs_snr")


def crps_rate_vs_snr():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for m in ["Diffusion+mu", "MeanFlow+mu", "MeanFlow-mu", "Diffusion-mu", "ConvLSTM"]:
        c, mk, ls = STYLE[m]
        ys = [DATA["stoch"][str(s)][m]["crps_rate"] for s in SNRS]
        ax.plot(SNRS, ys, color=c, marker=mk, linestyle=ls, lw=1.8, label=LABEL[m])
    ax.set_xlabel("Inference SNR (dB)"); ax.set_ylabel("CRPS of achievable rate  (↓)")
    ax.set_xticks(SNRS)
    ax.set_title("Downstream rate score vs SNR")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "crps_rate_vs_snr")


def coverage_vs_snr():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.axhline(0.9, color="k", ls=":", lw=1, label="nominal 0.90")
    for m in ["MeanFlow+mu", "Diffusion+mu", "MeanFlow-mu", "Diffusion-mu"]:
        c, mk, ls = STYLE[m]
        ys = [DATA["stoch"][str(s)][m]["cov90"] for s in SNRS]
        ax.plot(SNRS, ys, color=c, marker=mk, linestyle=ls, lw=1.8, label=LABEL[m])
    ax.set_xlabel("Inference SNR (dB)"); ax.set_ylabel("Coverage @ 90%")
    ax.set_xticks(SNRS); ax.set_ylim(0, 1)
    ax.set_title("Calibration vs SNR (higher=less overconfident)")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, "coverage_vs_snr")


if __name__ == "__main__":
    fig2_multiSNR(); fig3_nmse_vs_snr(); crps_rate_vs_snr(); coverage_vs_snr()
    print("saved fig2/fig3/crps_rate/coverage figures to", os.path.normpath(OUT))
