"""Computational-efficiency benchmark: diffusion DiU vs MeanFlow DiU.

Reports, for the full Nf-frame AUTOREGRESSIVE horizon, per model:
  * parameter count (encoder / generator / total) + on-disk / in-memory size
  * NFE per horizon  (encoder evals and generator/U-Net evals)
  * FLOPs per horizon  (via torch FlopCounterMode; falls back to `thop` or skips)
  * inference latency  (mean +/- std, ms) at batch size 1 and B, on GPU and/or CPU
  * peak GPU memory for one horizon prediction (MB)
  * a DiU sampling-step (NFE) sweep -> latency / FLOPs scaling for the Pareto view

No Sionna is required: inputs are random tensors of the correct shape. Weights do
not change FLOPs/params/latency, so checkpoints are OPTIONAL (pass --diu-ckpt /
--mf-ckpt only to benchmark the exact saved configs). Everything runs under
torch.no_grad (inference), matching deployment.

Usage (Alvis, A40):
    python -m scripts.benchmark_efficiency --batch 64 --repeats 30
    python -m scripts.benchmark_efficiency --batch 64 --cpu            # add CPU timing
    python -m scripts.benchmark_efficiency --diu-steps 1,2,3,5,10,20   # NFE sweep
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from mf_csi.config import Config
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.models.diu import DiUEncoder, DiUNet
from mf_csi.diffusion import make_scheduler, ddim_ar_predict
from mf_csi.inference import autoregressive_predict


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def flops_of(fn) -> "int | None":
    """Total FLOPs executed by fn(), via torch's FlopCounterMode. None if
    unavailable (older torch) -- we then try `thop`, else report None."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc:
            fn()
        return int(fc.get_total_flops())
    except Exception:
        return None


def time_ms(fn, device: str, warmup: int, repeats: int):
    """Return (mean_ms, std_ms) wall-clock for fn(), with CUDA sync + warmup."""
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


def peak_mem_mb(fn, device: str):
    if device != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def maybe_load(model, ckpt_path, keys, device):
    """Load weights if a checkpoint is given (weights don't affect timing/FLOPs,
    but this guarantees the exact saved architecture/config is benchmarked)."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return
    ck = torch.load(ckpt_path, map_location=device)
    for attr, key in keys.items():
        if key in ck:
            getattr(model, attr).load_state_dict(ck[key])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64, help="batched throughput size B")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--diu-steps", type=str, default="1,2,3,5,10,20",
                   help="DiU DDIM sampling-step sweep for the latency/FLOPs Pareto")
    p.add_argument("--mf-samples", type=int, default=1, help="MeanFlow 1-NFE draws per frame")
    p.add_argument("--cpu", action="store_true", help="also time on CPU (slow for DiU)")
    p.add_argument("--diu-ckpt", type=str, default=None)
    p.add_argument("--mf-ckpt", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs/bench")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    os.makedirs(args.out_dir, exist_ok=True)
    has_cuda = torch.cuda.is_available()
    dev = "cuda" if has_cuda else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "cpu"
    print(f"device={dev} ({gpu_name}) | Np={Np} Nf={Nf} grid={Nt}x{Nc} | "
          f"batch={args.batch} repeats={args.repeats}")

    # ---- build models ----
    enc_m = TemporalEncoder(cfg.encoder).to(dev).eval()
    gen_m = UNetGenerator(cfg.generator).to(dev).eval()
    enc_d = DiUEncoder(cfg.diu, in_channels=2).to(dev).eval()
    unet_d = DiUNet(cfg.diu, data_channels=2, image_size=Nc).to(dev).eval()
    scheduler = make_scheduler(cfg.diu)
    maybe_load(type("O", (), {"enc": enc_m, "gen": gen_m})(), args.mf_ckpt,
               {"enc": "ema_enc", "gen": "ema_gen"}, dev)
    maybe_load(type("O", (), {"enc": enc_d, "unet": unet_d})(), args.diu_ckpt,
               {"enc": "ema_enc", "unet": "ema_unet"}, dev)

    params = {
        "MeanFlow": {"encoder": count_params(enc_m), "generator": count_params(gen_m)},
        "DiU": {"encoder": count_params(enc_d), "generator": count_params(unet_d)},
    }
    for k, v in params.items():
        v["total"] = v["encoder"] + v["generator"]
        v["size_MB_fp32"] = v["total"] * 4 / 1024 ** 2

    # ---- horizon predict closures ----
    def mf_predict(B):
        past = torch.randn(B, Np, 2, Nt, Nc, device=dev)
        return lambda: autoregressive_predict(
            enc_m, gen_m, past, Nf,
            seed_std=cfg.inference.seed_std, num_samples=args.mf_samples)

    def diu_predict(B, steps):
        cfg.diu.sampling_steps = steps
        past = torch.randn(B, Np, 2, Nt, Nc, device=dev)
        return lambda: ddim_ar_predict(enc_d, unet_d, scheduler, past, Nf, cfg.diu)

    diu_step_list = [int(s) for s in args.diu_steps.split(",") if s.strip()]
    diu_main_steps = cfg.diu.sampling_steps if cfg.diu.sampling_steps in diu_step_list else diu_step_list[-1]

    # ---- NFE bookkeeping (per full horizon) ----
    nfe = {
        "MeanFlow": {"encoder_evals": Nf, "generator_evals": Nf * args.mf_samples},
        "DiU": {"encoder_evals": Nf, "unet_evals": Nf * diu_main_steps},
    }

    # ---- FLOPs per horizon ----
    with torch.no_grad():
        flops = {
            "MeanFlow": flops_of(mf_predict(1)),
            "DiU": flops_of(diu_predict(1, diu_main_steps)),
        }

    # ---- latency + memory ----
    results = {}
    with torch.no_grad():
        for name, mk in [("MeanFlow", lambda B: mf_predict(B)),
                         ("DiU", lambda B: diu_predict(B, diu_main_steps))]:
            r = {}
            m1, s1 = time_ms(mk(1), dev, args.warmup, args.repeats)
            mB, sB = time_ms(mk(args.batch), dev, args.warmup, args.repeats)
            r["latency_ms_b1"] = (m1, s1)
            r["latency_ms_bB"] = (mB, sB)
            r["throughput_sps_bB"] = args.batch / (mB / 1e3) if mB > 0 else None
            r["peak_mem_MB_bB"] = peak_mem_mb(mk(args.batch), dev)
            results[name] = r

        # DiU NFE sweep (latency @ B, FLOPs @ B=1)
        sweep = []
        for steps in diu_step_list:
            mB, sB = time_ms(diu_predict(args.batch, steps), dev, args.warmup, max(5, args.repeats // 2))
            fl = flops_of(diu_predict(1, steps))
            sweep.append({"nfe_per_frame": steps, "unet_evals": Nf * steps,
                          "latency_ms_bB": (mB, sB), "flops_horizon": fl})

        # optional CPU timing (B=1 only; DiU capped at a few steps to stay quick)
        cpu = None
        if args.cpu and dev == "cuda":
            enc_m.cpu(); gen_m.cpu(); enc_d.cpu(); unet_d.cpu()
            def mf_cpu():
                past = torch.randn(1, Np, 2, Nt, Nc)
                return lambda: autoregressive_predict(enc_m, gen_m, past, Nf,
                                                      seed_std=cfg.inference.seed_std,
                                                      num_samples=args.mf_samples)
            def diu_cpu(steps):
                cfg.diu.sampling_steps = steps
                past = torch.randn(1, Np, 2, Nt, Nc)
                return lambda: ddim_ar_predict(enc_d, unet_d, scheduler, past, Nf, cfg.diu)
            mmf, smf = time_ms(mf_cpu(), "cpu", 2, 5)
            mdu, sdu = time_ms(diu_cpu(diu_main_steps), "cpu", 1, 3)
            cpu = {"MeanFlow_ms_b1": (mmf, smf),
                   f"DiU@{diu_main_steps}nfe_ms_b1": (mdu, sdu)}
            enc_m.to(dev); gen_m.to(dev); enc_d.to(dev); unet_d.to(dev)

    # ---- report ----
    def fmt_flops(x):
        return "n/a" if x is None else f"{x/1e9:.2f} GFLOPs"

    print("\n==================== PARAMETERS ====================")
    for name in ("MeanFlow", "DiU"):
        v = params[name]
        print(f"{name:9s} | enc {v['encoder']/1e6:6.3f}M | gen {v['generator']/1e6:6.3f}M "
              f"| total {v['total']/1e6:6.3f}M ({v['size_MB_fp32']:.1f} MB fp32)")

    print("\n============ PER-HORIZON COMPUTE (Nf frames) ============")
    for name in ("MeanFlow", "DiU"):
        ge = nfe[name].get("generator_evals", nfe[name].get("unet_evals"))
        print(f"{name:9s} | enc evals {nfe[name]['encoder_evals']:3d} | "
              f"gen/unet evals {ge:3d} | FLOPs {fmt_flops(flops[name])}")

    print(f"\n=========== LATENCY & MEMORY (device={dev}) ===========")
    print(f"DiU benchmarked at {diu_main_steps} NFE/frame; MeanFlow at {args.mf_samples} draw/frame")
    for name in ("MeanFlow", "DiU"):
        r = results[name]
        mem = "n/a" if r["peak_mem_MB_bB"] is None else f"{r['peak_mem_MB_bB']:.0f} MB"
        tp = "n/a" if r["throughput_sps_bB"] is None else f"{r['throughput_sps_bB']:.0f}/s"
        print(f"{name:9s} | B=1 {r['latency_ms_b1'][0]:7.2f}+-{r['latency_ms_b1'][1]:.2f} ms "
              f"| B={args.batch} {r['latency_ms_bB'][0]:8.2f}+-{r['latency_ms_bB'][1]:.2f} ms "
              f"| {tp:>8s} | peak {mem}")

    print("\n=========== DiU NFE SWEEP (latency Pareto) ===========")
    for s in sweep:
        print(f"  {s['nfe_per_frame']:2d} NFE/frame ({s['unet_evals']:3d} evals) | "
              f"B={args.batch} {s['latency_ms_bB'][0]:8.2f} ms | FLOPs {fmt_flops(s['flops_horizon'])}")

    if cpu:
        print("\n=================== CPU LATENCY (B=1) ===================")
        for k, (m, s) in cpu.items():
            print(f"  {k}: {m:.1f}+-{s:.1f} ms")

    dump = {"device": gpu_name, "batch": args.batch, "Np": Np, "Nf": Nf,
            "params": params, "nfe_per_horizon": nfe, "flops_horizon": flops,
            "latency_memory": results, "diu_nfe_sweep": sweep, "cpu_latency": cpu,
            "diu_main_nfe": diu_main_steps, "mf_samples": args.mf_samples}
    out = os.path.join(args.out_dir, "efficiency.json")
    with open(out, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
