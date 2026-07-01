"""Tests for autoregressive inference + NMSE.

Run (needs torch):  python -m tests.test_inference
"""

import torch

from mf_csi.config import Config
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.inference import autoregressive_predict, predict_next_frame, nmse, nmse_db


def test_ar_shapes():
    cfg = Config()
    enc = TemporalEncoder(cfg.encoder)
    gen = UNetGenerator(cfg.generator)
    B = 2
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    past = torch.randn(B, Np, 2, Nt, Nc)
    pred = autoregressive_predict(enc, gen, past, Nf)
    assert pred.shape == (B, Nf, 2, Nt, Nc), pred.shape
    print(f"OK AR rollout -> {tuple(pred.shape)}")


def test_nmse_properties():
    cfg = Config()
    B, Nf, Nt, Nc = 4, cfg.data.num_future, cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    true = torch.randn(B, Nf, 2, Nt, Nc)
    # Identical prediction -> NMSE 0.
    ps, ov = nmse(true.clone(), true)
    assert torch.allclose(ov, torch.zeros(()), atol=1e-6), ov
    # Zero prediction -> NMSE 1 (error power == signal power).
    ps0, ov0 = nmse(torch.zeros_like(true), true)
    assert torch.allclose(ov0, torch.ones(()), atol=1e-6), ov0
    assert ps.shape == (Nf,)
    print(f"OK nmse: identical={ov.item():.2e}, zero-pred={ov0.item():.3f} ({nmse_db(ov0).item():.1f} dB)")


if __name__ == "__main__":
    test_ar_shapes()
    test_nmse_properties()
    print("\nAll inference tests passed.")
