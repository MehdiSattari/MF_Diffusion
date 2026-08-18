"""Diagnose WHY diffusion NMSE degrades with more DDIM steps.

Root-cause sweep for a single diffusion checkpoint (typically Diffusion+mu). For each
DDIM step count we vary the three suspect knobs and report accuracy AND calibration,
so we can tell which effect drives "more steps -> worse NMSE":

  * init     zeros    : start the sampler from 0 (the paper's 1-step MMSE trick), OR
             gaussian : x_T ~ N(0, I)  -- the correct multi-step initial condition
  * spacing  leading  : diffusers default step placement, OR
             trailing : the Stable-Diffusion few-step fix
  * eta      0.0      : deterministic PF-ODE (accuracy-optimal, single path), OR
             1.0      : stochastic (calibrated ensemble)

For every (steps, init, spacing, eta) cell we log:

  ensNMSE  : NMSE of the K-sample ENSEMBLE MEAN  (~ posterior mean = the MMSE/accuracy number)
  pNMSE    : mean per-sample NMSE                (a genuine draw; rewards calibrated spread less)
  cov90    : empirical coverage of the nominal 90% interval (calibration)
  crps     : CRPS (sharpness + calibration)

Interpretation:
  - If ensNMSE is ~flat across steps once init=gaussian/eta=0  -> accuracy is step-independent;
    the "degradation" was a wrong-IC / small-K artifact -> report the mean, no retrain.
  - If ensNMSE still degrades with steps under the corrected knobs -> parameterization issue
    -> the v-prediction retrain is the fix.

Channels are loaded from a pre-generated paired set (--load-batches) so every cell sees the
byte-identical channels. Reuses evaluate_uncertainty_2x2's data + metric helpers.

Usage (Alvis):
    python -m scripts.diagnose_diffsteps --diff-mu runs/ar_diffusion_muon_XXXX/ckpt_best.pt \
        --load-batches runs/eval_ch_sweep_seed0.pt --K 100 --snr 20 \
        --steps-list 1,3,10,20,50
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import torch

from mf_csi.config import Config, apply_generator_size
from mf_csi.diffusion import make_scheduler, corrupt_history
from mf_csi.diffusion_shared import ddim_ar_predict_shared
from mf_csi.inference import nmse_db
from mf_csi.uncertainty import crps, coverage, ensemble_mean_nmse, per_sample_nmse

# reuse the exact data + loading helpers from the 2x2 eval so channels are identical
from scripts.evaluate_uncertainty_2x2 import (
    set_all_seeds, raw_batches, to_batches, denorm_ens, load_gen)
from mf_csi.data.dataset import denormalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diff-mu", required=True, help="diffusion checkpoint to diagnose (residual/mu path)")
    p.add_argument("--use-mu", type=int, default=1, help="1=residual(mu) diffusion, 0=standard")
    p.add_argument("--prediction-type", type=str, default=None,
                   choices=["sample", "epsilon", "v_prediction"],
                   help="override; else read from ckpt meta, else config default ('sample')")
    p.add_argument("--gen-size", type=str, default=None,
                   choices=["xs", "small", "medium", "large", "xl"])
    p.add_argument("--steps-list", type=str, default="1,3,10,20,50")
    p.add_argument("--inits", type=str, default="zeros,gaussian")
    p.add_argument("--spacings", type=str, default="leading,trailing")
    p.add_argument("--etas", type=str, default="0.0,1.0")
    p.add_argument("--K", type=int, default=100)
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--n-samples", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load-batches", type=str, default=None)
    p.add_argument("--save-batches", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs/diffdiag")
    return p.parse_args()


@torch.no_grad()
def eval_cell(enc, gen, scheduler, cfg, batches, device, snr, K, use_mu):
    """NMSE at the FIRST and LAST AR prediction step (not the horizon average), so we can
    see rollout stability. Returns a dict with ens/per NMSE at step 1 and step Nf, plus
    horizon-averaged cov90 and CRPS."""
    ens_ps, per_ps, covs, crpss = [], [], [], []
    for b in batches:
        past, future = b["past"].to(device), b["future"].to(device)
        hist = corrupt_history(past, snr, snr)
        Nf = future.shape[1]
        s = torch.stack([ddim_ar_predict_shared(enc, gen, scheduler, hist, Nf, cfg.diu,
                                                 use_mu=use_mu) for _ in range(K)], dim=0)
        s = denorm_ens(s, b["stats"])                       # [K,B,Nf,2,Nt,Nc]
        y = denormalize(future, b["stats"])
        ens_ps.append(ensemble_mean_nmse(s, y)[0])          # per-step tensor [Nf]
        per_ps.append(per_sample_nmse(s, y)[0])             # per-step tensor [Nf]
        covs.append(coverage(s, y, 0.9)[1].item())
        crpss.append(crps(s, y)[0])
    ens = torch.stack(ens_ps).mean(0)                       # [Nf], linear
    per = torch.stack(per_ps).mean(0)                       # [Nf], linear
    return {"ens_first": nmse_db(ens[0]).item(), "ens_last": nmse_db(ens[-1]).item(),
            "per_first": nmse_db(per[0]).item(), "per_last": nmse_db(per[-1]).item(),
            "cov90": float(np.mean(covs)), "crps": float(torch.stack(crpss).mean(0).mean())}


def main():
    args = parse_args()
    cfg = Config()
    if args.gen_size:
        apply_generator_size(cfg.generator, args.gen_size)
    cfg.data.seed = args.seed
    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    use_mu = bool(args.use_mu)

    # parameterization: CLI > ckpt meta > config default
    ptype = args.prediction_type
    ck_meta = torch.load(args.diff_mu, map_location="cpu").get("meta", {})
    if ptype is None:
        ptype = ck_meta.get("prediction_type", cfg.diu.prediction_type)
    cfg.diu.prediction_type = ptype
    print(f"diagnosing {args.diff_mu} | prediction_type={ptype} | use_mu={use_mu} | "
          f"gen_size={args.gen_size or 'medium'} | K={args.K} | SNR={args.snr}", flush=True)

    if args.load_batches:
        batches = torch.load(args.load_batches, map_location="cpu")
        print(f"loaded {len(batches)} paired eval batches from {args.load_batches}", flush=True)
    else:
        batches = to_batches(raw_batches(cfg.data, args.n_samples, args.batch_size), cfg.data)
        if args.save_batches:
            torch.save(batches, args.save_batches)
            print(f"saved {len(batches)} eval batches to {args.save_batches}", flush=True)

    enc, gen = load_gen(args.diff_mu, cfg, device)

    steps_list = [int(x) for x in args.steps_list.split(",") if x.strip()]
    inits = args.inits.split(",")
    spacings = args.spacings.split(",")
    etas = [float(x) for x in args.etas.split(",") if x.strip()]

    rows = []
    header = (f"{'init':9s} {'spacing':9s} {'eta':4s} {'steps':>5s} | "
              f"{'ens@1':>7s} {'ens@N':>7s} {'p@1':>7s} {'p@N':>7s} {'cov90':>6s} {'CRPS':>7s}")
    print("\n(ens@1/ens@N = ensemble-mean NMSE at first/last AR step; p = per-sample)")
    print(header); print("-" * len(header))
    for init in inits:
        cfg.diu.deterministic_init = (init == "zeros")
        for spacing in spacings:
            cfg.diu.timestep_spacing = spacing
            for eta in etas:
                cfg.diu.ddim_eta = eta
                # zeros-init + eta=0 is a single deterministic path -> K=1 is enough
                Kc = 1 if (init == "zeros" and eta == 0.0) else args.K
                for st in steps_list:
                    cfg.diu.sampling_steps = st
                    scheduler = make_scheduler(cfg.diu)
                    set_all_seeds(args.seed)             # identical sampling draws per cell
                    m = eval_cell(enc, gen, scheduler, cfg, batches, device, args.snr, Kc, use_mu)
                    print(f"{init:9s} {spacing:9s} {eta:<4.1f} {st:5d} | "
                          f"{m['ens_first']:7.2f} {m['ens_last']:7.2f} {m['per_first']:7.2f} "
                          f"{m['per_last']:7.2f} {m['cov90']:6.3f} {m['crps']:7.3f}  (K={Kc})", flush=True)
                    rows.append({"init": init, "spacing": spacing, "eta": eta, "steps": st,
                                 "K": Kc, **m})

    out = os.path.join(args.out_dir, "diffsteps_diagnosis.json")
    json.dump({"ckpt": args.diff_mu, "prediction_type": ptype, "use_mu": use_mu,
               "snr": args.snr, "rows": rows}, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
