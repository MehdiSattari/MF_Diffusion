"""Controlled seq2seq comparison: MeanFlow vs diffusion vs ConvLSTM (JointRegressor).

All three evaluated on ONE shared set of Sionna channels, each normalized in its own
global-std stats then DENORMALIZED to the physical channel for a fair NMSE. Only the
objective differs (same backbone for MeanFlow/diffusion). Produces one physical-space
NMSE-vs-step figure + JSON.

Usage (Alvis):
    python -m scripts.evaluate_seq2seq --size large \
        --mf-ckpt runs/s2s_mf_L/ckpt_best.pt --diff-ckpt runs/s2s_di_L/ckpt_best.pt \
        --reg-ckpt runs/s2s_reg/ckpt_best.pt --snrs 0,10,20 --clean
"""

from __future__ import annotations

import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mf_csi.config import Config
from mf_csi.data.sionna_cdl import CDLChannelGenerator
from mf_csi.data.dataset import _normalize, _split_batch, denormalize
from mf_csi.models import UNetGenerator, JointRegressor
from mf_csi.diffusion import make_scheduler, corrupt_history
from mf_csi.seq2seq import (seq2seq_unet_config, meanflow_seq2seq_predict,
                            diffusion_seq2seq_predict)
from mf_csi.inference import nmse, nmse_db


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mf-ckpt", type=str, default=None)
    p.add_argument("--diff-ckpt", type=str, default=None)
    p.add_argument("--reg-ckpt", type=str, default=None)
    p.add_argument("--size", default="large", choices=["small", "large"])
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--snrs", type=str, default="20")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--out-dir", type=str, default="runs/s2s_compare")
    return p.parse_args()


def build_unet(cfg, size, device):
    ucfg = seq2seq_unet_config(cfg.data.num_past, cfg.data.num_future,
                               ch_mult=(2 if size == "large" else 1),
                               num_res_blocks=(2 if size == "large" else 1),
                               use_attention=(size == "large"))
    return UNetGenerator(ucfg).to(device)


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    return ck["net"], ck.get("global_ab"), ck.get("meta", {})


def raw_batches(cfg, n, bs):
    gen = CDLChannelGenerator(cfg); out = []; rem = n
    while rem > 0:
        b = min(bs, rem); out.append(gen.generate(b)); rem -= b
    return out


def to_batches(raw, cfg, global_ab):
    out = []
    for c in raw:
        c_n, stats = _normalize(c, "global_std", global_ab)
        out.append(_split_batch(c_n, cfg, stats))
    return out


@torch.no_grad()
def per_step(pred_fn, batches, device, snr, cfg):
    ps_sum = None
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = past if snr is None else corrupt_history(past, snr, snr)
        pred = pred_fn(hist, future.shape[1])
        pr = denormalize(pred, b["stats"]); fu = denormalize(future, b["stats"])
        ps, _ = nmse(pr, fu)
        ps_sum = ps if ps_sum is None else ps_sum + ps
    return ps_sum / len(batches)


def main():
    args = parse_args()
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    scheduler = make_scheduler(cfg.diu)
    cfg.inference.seed_std = cfg.meanflow.source_std

    print(f"generating shared eval set: {args.n_samples} samples ...")
    raw = raw_batches(cfg.data, args.n_samples, args.batch_size)

    models = {}   # name -> (pred_fn, batches)
    if args.mf_ckpt:
        sd, ab, _ = load(args.mf_ckpt, device); net = build_unet(cfg, args.size, device)
        net.load_state_dict(sd); net.eval()
        models["MeanFlow"] = ((lambda h, Nf, n=net: meanflow_seq2seq_predict(n, h, Nf, seed_std=cfg.inference.seed_std)),
                              to_batches(raw, cfg.data, ab))
    if args.diff_ckpt:
        sd, ab, _ = load(args.diff_ckpt, device); net = build_unet(cfg, args.size, device)
        net.load_state_dict(sd); net.eval()
        models["Diffusion"] = ((lambda h, Nf, n=net: diffusion_seq2seq_predict(n, scheduler, h, Nf, cfg.diu)),
                               to_batches(raw, cfg.data, ab))
    if args.reg_ckpt:
        sd, ab, _ = load(args.reg_ckpt, device); net = JointRegressor(cfg.regression).to(device)
        net.load_state_dict(sd); net.eval()
        models["ConvLSTM"] = ((lambda h, Nf, n=net: n(h)), to_batches(raw, cfg.data, ab))

    snrs = [float(s) for s in args.snrs.split(",") if s.strip()]
    conditions = ([(None, "clean")] if args.clean else []) + [(s, f"{s:g} dB") for s in snrs]
    palette = {"Diffusion": "#1f77b4", "MeanFlow": "#d62728", "ConvLSTM": "#2ca02c"}
    styles = ["-", "--", "-.", ":"]

    dump = {"steps": None, "series": []}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    print("\n=== physical-space NMSE (dB) ===")
    for k, (snr, tag) in enumerate(conditions):
        ls = styles[k % len(styles)]
        for name, (pred_fn, batches) in models.items():
            ps = per_step(pred_fn, batches, device, snr, cfg)
            db = [round(nmse_db(v).item(), 3) for v in ps]
            avg = round(nmse_db(ps.mean()).item(), 3)
            dump["steps"] = list(range(1, len(ps) + 1))
            dump["series"].append({"model": name, "condition": tag, "nmse_db": db, "avg_nmse_db": avg})
            ax.plot(range(1, len(ps) + 1), db, marker="o", ms=4, color=palette[name],
                    linestyle=ls, label=f"{name} ({tag})")
            print(f"  {name:10s} ({tag:>6s}) | step1 {db[0]:7.2f} | step{len(db)} {db[-1]:7.2f} | avg {avg:7.2f}")

    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("prediction step"); ax.set_ylabel("NMSE (dB), physical space")
    ax.set_title(f"Seq2seq: MeanFlow vs Diffusion vs ConvLSTM ({args.size} backbone)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"nmse_vs_step_seq2seq_{args.size}.png")
    js = os.path.join(args.out_dir, f"nmse_vs_step_seq2seq_{args.size}.json")
    fig.savefig(png, dpi=150); plt.close(fig)
    with open(js, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nsaved: {png}\n       {js}")


if __name__ == "__main__":
    main()
