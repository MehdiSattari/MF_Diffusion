"""Report ensemble spread, skill (RMSE), and spread/skill vs SNR from sweep run JSONs.

This is the correct instrument for the "is the uncertainty adaptive?" question: coverage
entangles spread with bias and distribution shape, whereas spread and skill are read off
directly. Scans the stochastic SNR-sweep run directories, extracts horizon-averaged
spread, skill, spread/skill, and coverage per model, prints a table, and saves a 3-panel
figure (spread / skill / spread-skill vs SNR).

Usage (Alvis, native FS):
    python -m scripts.spread_vs_snr --glob 'runs/uq2x2_snr*_stoch_*' --out figures/spread_vs_snr
"""
from __future__ import annotations
import argparse, glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["MeanFlow+mu", "Diffusion+mu", "MeanFlow-mu", "Diffusion-mu"]
STYLE = {"MeanFlow+mu": ("#d62728", "o"), "Diffusion+mu": ("#1f77b4", "s"),
         "MeanFlow-mu": ("#f0997b", "o"), "Diffusion-mu": ("#85b7eb", "s"),
         "MeanFlowCol+mu": ("#9467bd", "D"), "Gauss(mu,R)": ("#8c564b", "x")}
LAB = {"MeanFlow+mu": "MeanFlow $+\\mu$", "Diffusion+mu": "Diffusion $+\\mu$",
       "MeanFlow-mu": "MeanFlow $-\\mu$", "Diffusion-mu": "Diffusion $-\\mu$",
       "MeanFlowCol+mu": "MeanFlow $+\\mu$ (colored)", "Gauss(mu,R)": "$\\mathcal{N}(\\mu,R)$"}


def snr_of(path):
    m = re.search(r"_snr(\d+)_", path)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/uq2x2_snr*_stoch_*")
    ap.add_argument("--out", default="figures/spread_vs_snr")
    ap.add_argument("--models", nargs="*", default=["MeanFlow+mu", "Diffusion+mu"])
    args = ap.parse_args()

    # newest run per SNR
    byid = {}
    for d in sorted(glob.glob(args.glob)):
        if "_ds" in d:            # skip the DDIM-step sweep dirs
            continue
        j = os.path.join(d, "uncertainty_2x2.json")
        s = snr_of(d)
        if s is None or not os.path.isfile(j):
            continue
        byid[s] = j               # later (sorted) wins -> newest timestamp
    snrs = sorted(byid)
    if not snrs:
        print(f"no run JSONs matched {args.glob}"); return
    for s in snrs:                       # show exactly which run dir is used per SNR
        d = json.load(open(byid[s]))
        print(f"[SNR {s:>2}] {os.path.dirname(byid[s])}  models={sorted(k for k in d if 'nmse' in d[k])}")

    data = {m: {"snr": [], "spread": [], "skill": [], "ratio": [], "cov": []} for m in args.models}
    print(f"{'SNR':>4} | {'model':13} | {'spread':>8} | {'skill':>8} | {'sp/sk':>6} | {'cov90':>6}")
    for s in snrs:
        d = json.load(open(byid[s]))
        for m in args.models:
            v = d.get(m)
            if not v or "spread" not in v:
                continue
            sp = sum(v["spread"]) / len(v["spread"])
            sk = sum(v["skill"]) / len(v["skill"])
            data[m]["snr"].append(s); data[m]["spread"].append(sp)
            data[m]["skill"].append(sk); data[m]["ratio"].append(sp / sk)
            data[m]["cov"].append(v.get("cov0.9", float("nan")))
            print(f"{s:>4} | {m:13} | {sp:8.4f} | {sk:8.4f} | {sp/sk:6.3f} | {v.get('cov0.9',0):6.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for m in args.models:
        D = data[m]
        if not D["snr"]:            # model absent from these runs -> skip (no crash)
            print(f"note: '{m}' not present in the matched runs; skipping.")
            continue
        c, mk = STYLE.get(m, ("#333333", "o"))
        ax[0].plot(D["snr"], D["spread"], marker=mk, color=c, label=LAB[m])
        ax[1].plot(D["snr"], D["skill"], marker=mk, color=c, label=LAB[m])
        ax[2].plot(D["snr"], D["ratio"], marker=mk, color=c, label=LAB[m])
    ax[0].set_ylabel("ensemble spread"); ax[1].set_ylabel("skill (RMSE)")
    ax[2].set_ylabel("spread / skill"); ax[2].axhline(1.0, ls=":", color="k", lw=1)
    for a in ax:
        a.set_xlabel("inference SNR (dB)"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{args.out}.{e}", dpi=200, bbox_inches="tight")
    print(f"\nsaved {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
