"""Paper-style (Fig. 2) NMSE-vs-prediction-step figure for the DETERMINISTIC-DDIM run.

Reproduces the visual format of Fig. 2 in "CSI Prediction Using Diffusion Models"
(NMSE in dB versus prediction step) using the deterministic-DDIM (eta=0) 2x2 results.
Standalone: per-step NMSE (dB) is embedded so the figure regenerates without the run
directory. Regenerate the numbers with scripts/evaluate_uncertainty_2x2.py --ddim-eta 0.

Saves PNG + PDF to figures/deterministic_ddim/. NOT added to the paper.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Deterministic DDIM (eta=0), paired eval, inference SNR = 20 dB, per-step NMSE [dB].
NMSE_20DB = {
    "Diffusion ($+\\mu$)": [-21.981, -19.188, -16.638, -14.464, -12.666, -11.177, -9.925, -8.832, -7.850, -6.960],
    "Diffusion ($-\\mu$)": [-10.073, -9.239, -7.435, -5.744, -4.513, -3.690, -3.074, -2.595, -2.160, -1.792],
    "MeanFlow ($+\\mu$)":  [-23.009, -20.215, -17.490, -15.055, -12.939, -11.123, -9.557, -8.183, -6.961, -5.876],
    "ConvLSTM":            [-20.644, -17.648, -14.953, -12.701, -10.851, -9.328, -8.057, -6.978, -6.056, -5.266],
}
STYLE = {
    "Diffusion ($+\\mu$)": ("#1f77b4", "s", "-"),
    "Diffusion ($-\\mu$)": ("#85b7eb", "s", "--"),
    "MeanFlow ($+\\mu$)":  ("#d62728", "o", "-"),
    "ConvLSTM":            ("#2ca02c", "^", ":"),
}


def main():
    outdir = os.path.join(os.path.dirname(__file__), "..", "figures", "deterministic_ddim")
    os.makedirs(outdir, exist_ok=True)
    steps = list(range(1, 11))

    plt.rcParams.update({"font.size": 12, "axes.grid": True,
                         "grid.alpha": 0.3, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for name, ys in NMSE_20DB.items():
        c, m, ls = STYLE[name]
        ax.plot(steps, ys, color=c, marker=m, linestyle=ls, markersize=5,
                linewidth=1.8, label=name)
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("NMSE (dB)")
    ax.set_xlim(1, 10); ax.set_xticks(steps)
    ax.set_title("Deterministic DDIM ($\\zeta{=}0$), inference SNR $=20$ dB")
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"nmse_vs_step_det_20dB.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved nmse_vs_step_det_20dB.{png,pdf} to", os.path.normpath(outdir))


if __name__ == "__main__":
    main()
