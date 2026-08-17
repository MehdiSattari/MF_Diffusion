"""Per-step (single CSI, B=1) inference-complexity analysis.

The operationally-relevant unit of AR CSI prediction is ONE prediction step for ONE user
(B=1). Its cost decomposes as

    per-step cost = 1 recurrent encoder step  +  N_NFE generator evaluations,

where the encoder is advanced ONE frame (O(1)) via the ConvLSTM recurrence -- NOT by
reprocessing the whole history. MeanFlow uses N_NFE=1; diffusion uses N_NFE>1. This script
measures each component at B=1 on the given device and reports:

  * one recurrent encoder step (stateful) vs a full-history re-encode (the naive cost)
  * one generator (U-Net) evaluation
  * per-step total = encoder-step + N_NFE * generator, for a sweep of N_NFE
  * FLOPs for each component

It also verifies the stateful step is numerically identical to re-encoding the history.

Usage (Alvis):
    python -m scripts.benchmark_perstep --gen-size medium --nfe-sweep 1,3,10,20,50 --repeats 100
"""
from __future__ import annotations
import argparse, json, os, statistics, time
import torch

from mf_csi.config import Config, apply_generator_size
from mf_csi.models import TemporalEncoder, UNetGenerator


def time_ms(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        else:
            t0 = time.perf_counter(); fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(ts), (statistics.pstdev(ts) if len(ts) > 1 else 0.0)


def flops_of(fn):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc:
            fn()
        return int(fc.get_total_flops())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-size", type=str, default="medium",
                    choices=["xs", "small", "medium", "large", "xl"])
    ap.add_argument("--nfe-sweep", type=str, default="1,3,10,20,50")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default="runs/bench")
    args = ap.parse_args()

    cfg = Config()
    apply_generator_size(cfg.generator, args.gen_size)
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"

    enc = TemporalEncoder(cfg.encoder).to(dev).eval()
    gen = UNetGenerator(cfg.generator).to(dev).eval()
    nfe_list = [int(s) for s in args.nfe_sweep.split(",") if s.strip()]

    past = torch.randn(1, Np, 2, Nt, Nc, device=dev)
    frame = torch.randn(1, 2, Nt, Nc, device=dev)
    r = torch.zeros(1, device=dev); t = torch.ones(1, device=dev)

    with torch.no_grad():
        # --- correctness: stateful step == re-encode(history + frame) ---
        states, _ = enc.warmup(past)
        (z_step, mu_step), _ = enc.step(frame, states, return_mu=True)
        hist2 = torch.cat([past, frame.unsqueeze(1)], dim=1)
        z_full, mu_full = enc(hist2, return_mu=True)
        z_err = (z_step - z_full).abs().max().item()
        mu_err = (mu_step - mu_full).abs().max().item() if mu_step is not None else 0.0

        z, _ = enc.warmup(past)                                    # for timing closures
        st, _ = enc.warmup(past)
        (z0, _), _ = enc.step(frame, st, return_mu=True)

        def enc_step():   return enc.step(frame, st, return_mu=True)
        def enc_full():   return enc(torch.cat([past, frame.unsqueeze(1)], 1), return_mu=True)
        def gen_eval():   return gen(frame, z0, r, t)

        enc_step_ms = time_ms(enc_step, dev, args.warmup, args.repeats)
        enc_full_ms = time_ms(enc_full, dev, args.warmup, args.repeats)
        gen_ms = time_ms(gen_eval, dev, args.warmup, args.repeats)

        fl_enc_step = flops_of(enc_step); fl_enc_full = flops_of(enc_full); fl_gen = flops_of(gen_eval)

    e1 = enc_step_ms[0]; g1 = gen_ms[0]
    per_step = [{"nfe": n, "latency_ms": e1 + n * g1, "gen_ms": n * g1,
                 "flops": (None if (fl_enc_step is None or fl_gen is None) else fl_enc_step + n * fl_gen)}
                for n in nfe_list]

    npar = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in gen.parameters())
    fmtf = lambda x: "n/a" if x is None else f"{x/1e6:.2f} MFLOPs"
    print(f"device={dev} ({name}) | gen_size={args.gen_size} | params={npar/1e6:.3f}M | B=1, single step")
    print(f"\ncorrectness (stateful step vs re-encode history): max|dz|={z_err:.2e}  max|dmu|={mu_err:.2e}")
    print("\n=================== PER-STEP COMPONENTS (B=1) ===================")
    print(f"  encoder step (stateful, O(1))     : {e1:7.3f} +- {enc_step_ms[1]:.3f} ms  | {fmtf(fl_enc_step)}")
    print(f"  encoder re-encode history (naive) : {enc_full_ms[0]:7.3f} +- {enc_full_ms[1]:.3f} ms  | {fmtf(fl_enc_full)}"
          f"   ({enc_full_ms[0]/max(e1,1e-9):.1f}x the stateful step)")
    print(f"  one generator (U-Net) evaluation  : {g1:7.3f} +- {gen_ms[1]:.3f} ms  | {fmtf(fl_gen)}")
    print("\n============ PER-STEP TOTAL = enc_step + NFE x generator ============")
    for p in per_step:
        tag = "MeanFlow" if p["nfe"] == 1 else "Diffusion"
        sp = per_step[0]["latency_ms"]
        print(f"  {p['nfe']:2d} NFE [{tag:9s}] : {p['latency_ms']:7.3f} ms  ({p['latency_ms']/sp:.1f}x vs 1-NFE)  | {fmtf(p['flops'])}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"perstep_{args.gen_size}.json")
    json.dump({"device": name, "gen_size": args.gen_size, "params_M": npar / 1e6,
               "enc_step_ms": enc_step_ms, "enc_full_ms": enc_full_ms, "gen_ms": gen_ms,
               "flops": {"enc_step": fl_enc_step, "enc_full": fl_enc_full, "gen": fl_gen},
               "per_step": per_step, "correctness": {"dz": z_err, "dmu": mu_err}}, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
