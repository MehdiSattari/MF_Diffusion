"""Overlay per-step (per prediction horizon) NMSE curves for several models.

Built to compare MeanFlow vs diffusion at different generator sizes (e.g. medium vs xl)
on ONE shared, paired channel set. Each --model is "label|objective|gen_size|ckpt":

    objective : meanflow | diffusion   (both use the informative +mu / residual path)
    gen_size  : xs|small|medium|large|xl  (MUST match how the ckpt was trained)

For each model it rolls out the AR prediction, computes the ensemble-mean NMSE per
prediction step (accuracy), and plots all curves together (MeanFlow=red, Diffusion=blue;
medium=solid, xl=dashed by default). Diffusion uses the corrected sampler (trailing
spacing, deterministic by default) so the comparison is on the good operating point.

Usage (Alvis), once all four checkpoints exist:
    python -m scripts.plot_perstep_nmse --load-batches runs/eval_ch_sweep_seed0.pt \
        --snr 20 --K 32 --diff-steps 20 --spacing trailing --ddim-eta 0 --init gaussian \
        --model "MeanFlow (medium)|meanflow|medium|runs/mf_6913678/ckpt_best.pt" \
        --model "MeanFlow (xl)|meanflow|xl|runs/ar_meanflow_muon_gsxl_XXXX/ckpt_best.pt" \
        --model "Diffusion (medium)|diffusion|medium|runs/ar_diffusion_muon_YYYY/ckpt_best.pt" \
        --model "Diffusion (xl)|diffusion|xl|runs/ar_diffusion_muon_gsxl_ZZZZ/ckpt_best.pt" \
        --out-dir runs/perstep_xl_vs_medium
"""
from __future__ import annotations

import argparse, json, os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mf_csi.config import Config, apply_generator_size
from mf_csi.diffusion import make_scheduler, corrupt_history
from mf_csi.diffusion_shared import ddim_ar_predict_shared
from mf_csi.inference import autoregressive_predict, nmse_db
from mf_csi.uncertainty import ensemble_mean_nmse
from mf_csi.data.dataset import denormalize
from scripts.evaluate_uncertainty_2x2 import set_all_seeds, raw_batches, to_batches, denorm_ens, load_gen

COLOR = {"meanflow": "#d62728", "diffusion": "#1f77b4"}
STYLE = {"medium": ("-", "o"), "xl": ("--", "s"),
         "large": ("-.", "^"), "small": (":", "v"), "xs": (":", "x")}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True,
                   help='"label|objective|gen_size|ckpt"; repeatable')
    p.add_argument("--load-batches", type=str, default="runs/eval_ch_sweep_seed0.pt")
    p.add_argument("--save-batches", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--K", type=int, default=32, help="ensemble size for the mean-NMSE accuracy curve")
    p.add_argument("--diff-steps", type=int, default=20)
    p.add_argument("--spacing", type=str, default="trailing", choices=["leading", "trailing", "linspace"])
    p.add_argument("--ddim-eta", type=float, default=0.0)
    p.add_argument("--init", type=str, default="gaussian", choices=["gaussian", "zeros"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="runs/perstep_cmp")
    return p.parse_args()


@torch.no_grad()
def per_step_nmse_db(label, obj, size, ckpt, batches, device, args):
    cfg = Config()
    apply_generator_size(cfg.generator, size)
    cfg.inference.seed_std = cfg.meanflow.source_std
    cfg.diu.sampling_steps = args.diff_steps
    cfg.diu.ddim_eta = args.ddim_eta
    cfg.diu.timestep_spacing = args.spacing
    cfg.diu.deterministic_init = (args.init == "zeros")
    # read prediction_type from ckpt meta so a v-pred model samples correctly
    meta = torch.load(ckpt, map_location="cpu").get("meta", {})
    cfg.diu.prediction_type = meta.get("prediction_type", cfg.diu.prediction_type)
    enc, gen = load_gen(ckpt, cfg, device)
    scheduler = make_scheduler(cfg.diu) if obj == "diffusion" else None
    acc = []
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = corrupt_history(past, args.snr, args.snr)
        Nf = future.shape[1]
        if obj == "meanflow":
            s = torch.stack([autoregressive_predict(enc, gen, hist, Nf,
                             seed_std=cfg.inference.seed_std, num_samples=1, use_mu=True)
                             for _ in range(args.K)], dim=0)
        else:
            s = torch.stack([ddim_ar_predict_shared(enc, gen, scheduler, hist, Nf, cfg.diu,
                             use_mu=True) for _ in range(args.K)], dim=0)
        s = denorm_ens(s, b["stats"]); y = denormalize(future, b["stats"])
        acc.append(ensemble_mean_nmse(s, y)[0])            # per-step [Nf], linear
    per = torch.stack(acc).mean(0)                          # [Nf]
    db = nmse_db(per).tolist()
    print(f"[{label:22s}] per-step NMSE (dB): " + " ".join(f"{v:.2f}" for v in db), flush=True)
    return db


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    cfg0 = Config(); cfg0.data.seed = args.seed
    set_all_seeds(args.seed)
    if os.path.isfile(args.load_batches):
        batches = torch.load(args.load_batches, map_location="cpu")
        print(f"loaded {len(batches)} paired eval batches from {args.load_batches}")
    else:
        batches = to_batches(raw_batches(cfg0.data, args.n_samples, args.batch_size), cfg0.data)
        if args.save_batches:
            torch.save(batches, args.save_batches); print(f"saved batches to {args.save_batches}")

    print(f"sampler: obj-diffusion uses spacing={args.spacing}, eta={args.ddim_eta}, "
          f"init={args.init}, steps={args.diff_steps}; K={args.K}; SNR={args.snr}")

    curves = {}
    for spec in args.model:
        parts = spec.split("|")
        if len(parts) != 4:
            raise ValueError(f'--model must be "label|objective|gen_size|ckpt", got: {spec}')
        label, obj, size, ckpt = [x.strip() for x in parts]
        if not os.path.isfile(ckpt):
            print(f"  [skip] {label}: ckpt not found: {ckpt}"); continue
        db = per_step_nmse_db(label, obj, size, ckpt, batches, device, args)
        curves[label] = {"objective": obj, "gen_size": size, "nmse_db": db}

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for label, c in curves.items():
        steps = list(range(1, len(c["nmse_db"]) + 1))
        ls, mk = STYLE.get(c["gen_size"], ("-", "o"))
        ax.plot(steps, c["nmse_db"], linestyle=ls, marker=mk, ms=4,
                color=COLOR.get(c["objective"], "black"), label=label)
    ax.set_xlabel("prediction step"); ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"Per-step NMSE: generator size (medium vs xl), {args.snr:g} dB")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"perstep_nmse.{ext}"), dpi=150)
    plt.close(fig)
    json.dump({"args": vars(args), "curves": curves},
              open(os.path.join(args.out_dir, "perstep_nmse.json"), "w"), indent=2)
    print(f"\nsaved perstep_nmse.png/.pdf/.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
