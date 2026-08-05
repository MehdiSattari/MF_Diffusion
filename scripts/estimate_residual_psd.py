"""Estimate the normalized angle-delay residual PSD for the colored source prior.

Loads a trained encoder, generates Sionna CDL channels, computes the residual r = Y - mu
of the encoder's point estimate on the FIRST future frame, and accumulates the normalized
2D-DFT power spectral density (circulant approx of the residual covariance R). Saves the
PSD [Nt, Nc] to a file for use as the colored source in training/inference.

Usage (Alvis):
    python -m scripts.estimate_residual_psd --enc runs/mf_6913678/ckpt_best.pt \
        --n-samples 4096 --out runs/residual_psd.pt
"""
from __future__ import annotations
import argparse
import torch

from mf_csi.config import Config
from mf_csi.data.sionna_cdl import CDLChannelGenerator
from mf_csi.data.dataset import _normalize, _split_batch
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.diffusion import corrupt_history
from mf_csi.colored_prior import to_complex


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--enc", required=True, help="checkpoint with a trained encoder (ema_enc)")
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--snr", type=float, default=20.0, help="history corruption SNR for mu")
    p.add_argument("--per-step", action="store_true", help="also save a per-horizon-step PSD stack")
    p.add_argument("--out", type=str, default="runs/residual_psd.pt")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.enc, map_location=device)
    enc = TemporalEncoder(cfg.encoder).to(device)
    enc.load_state_dict(ck["ema_enc"]); enc.eval()

    gen = CDLChannelGenerator(cfg.data)
    acc = None; acc_steps = None; n = 0
    with torch.no_grad():
        rem = args.n_samples
        while rem > 0:
            b = min(args.batch_size, rem); rem -= b
            c = gen.generate(b)
            c_n, stats = _normalize(c, "std", None)
            batch = _split_batch(c_n, cfg.data, stats)
            past, future = batch["past"].to(device), batch["future"].to(device)
            hist = corrupt_history(past, args.snr, args.snr)
            _, mu = enc(hist, return_mu=True)                    # mu ~ E[Y0 | history]
            r = future[:, 0] - mu                                # residual on first frame [b,2,Nt,Nc]
            rf = torch.fft.fft2(to_complex(r), norm="ortho")     # [b,Nt,Nc]
            psd = (rf.abs() ** 2).sum(dim=0)                     # accumulate
            acc = psd if acc is None else acc + psd
            n += b
            if args.per_step:
                # residual per horizon step needs an AR rollout; skipped in v1 (global PSD).
                pass
    psd = (acc / n)
    psd = psd / psd.mean().clamp_min(1e-8)                       # normalize to unit mean power
    torch.save({"psd": psd.cpu(), "n": n, "snr": args.snr, "enc": args.enc}, args.out)
    top = torch.sort(psd.flatten(), descending=True).values
    frac = float(top[:5].sum() / psd.sum())
    print(f"saved normalized PSD [{tuple(psd.shape)}] to {args.out} from {n} samples | "
          f"top-5 angle-delay bins hold {100*frac:.1f}% of residual energy (isotropic ~ {100*5/psd.numel():.1f}%)")


if __name__ == "__main__":
    main()
