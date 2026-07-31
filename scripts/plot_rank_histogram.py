"""PIT / verification-rank histograms from a 2x2 eval JSON (paper-style).

Reads an uncertainty_2x2.json that contains per-model "rank_hist" (written by
scripts/evaluate_uncertainty_2x2.py) and renders rank histograms. A calibrated ensemble
is FLAT (bars near the dashed uniform line); an overconfident/under-dispersed one is
U-shaped (mass piling at the ends). Saves to figures/deterministic_ddim/.

Usage:
    python scripts/plot_rank_histogram.py runs/uq2x2_snr20_stoch_XXXX/uncertainty_2x2.json
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"MeanFlow+mu": "MeanFlow ($+\\mu$)", "Diffusion+mu": "Diffusion ($+\\mu$)",
         "MeanFlow-mu": "MeanFlow ($-\\mu$)", "Diffusion-mu": "Diffusion ($-\\mu$)"}
COLOR = {"MeanFlow+mu": "#d62728", "Diffusion+mu": "#1f77b4",
         "MeanFlow-mu": "#f0997b", "Diffusion-mu": "#85b7eb"}
ORDER = ["MeanFlow+mu", "Diffusion+mu", "MeanFlow-mu", "Diffusion-mu"]


def main(path):
    data = json.load(open(path))
    models = [m for m in ORDER if m in data and "rank_hist" in data[m]]
    if not models:
        print("no rank_hist in", path, "- re-run eval with the updated script."); return
    outdir = os.path.join(os.path.dirname(__file__), "..", "figures", "deterministic_ddim")
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        h = data[m]["rank_hist"]; nb = len(h)
        u = data[m].get("rank_uniformity", sum(abs(x - 1.0 / nb) for x in h))
        ax.bar(range(nb), h, color=COLOR[m], edgecolor="black", linewidth=0.4, width=0.9)
        ax.axhline(1.0 / nb, color="k", ls="--", lw=1)
        ax.set_title(f"{LABEL[m]}\n(rankU={u:.2f})", fontsize=10)
        ax.set_xlabel("PIT bin"); ax.set_xticks([0, nb // 2, nb - 1])
    axes[0].set_ylabel("frequency")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"rank_histogram.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved rank_histogram.{png,pdf} to", os.path.normpath(outdir))
    for m in models:
        print(f"  {m:13s} rankU={data[m].get('rank_uniformity', float('nan')):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "uncertainty_2x2.json")
