"""Shape, gradient, and JVP tests for the U-Net generator.

The JVP test is important: the MeanFlow objective (Step 4) computes d/dt u via a
Jacobian-vector product with tangent (v, 0, 1) on inputs (h, r, t). If that works
here, Step 4 is de-risked.

Run (needs torch):  python -m tests.test_unet
"""

import torch
import torch.func as func

from mf_csi.config import Config
from mf_csi.models import UNetGenerator


def test_generator_shapes_and_grad():
    cfg = Config()
    gen = UNetGenerator(cfg.generator)
    B = 3
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    h = torch.randn(B, cfg.generator.in_channels, Nt, Nc)
    z = torch.randn(B, cfg.generator.cond_channels, Nt, Nc)
    r, t = torch.rand(B), torch.rand(B)
    u = gen(h, z, r, t)
    assert u.shape == (B, cfg.generator.out_channels, Nt, Nc), u.shape
    u.pow(2).mean().backward()
    n = sum(p.numel() for p in gen.parameters())
    print(f"OK generator u {tuple(u.shape)} | params {n/1e6:.2f}M")


def test_generator_jvp():
    cfg = Config()
    gen = UNetGenerator(cfg.generator).eval()
    B = 2
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    h = torch.randn(B, 2, Nt, Nc)
    z = torch.randn(B, cfg.generator.cond_channels, Nt, Nc)
    r, t = torch.rand(B), torch.rand(B)
    v = torch.randn_like(h)                       # tangent for h (instant. velocity)

    def f(h_, r_, t_):
        return gen(h_, z, r_, t_)                 # Z held fixed (tangent 0)

    primal, dudt = func.jvp(f, (h, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    assert primal.shape == dudt.shape == (B, 2, Nt, Nc)
    assert torch.isfinite(dudt).all()
    print(f"OK jvp d/dt u {tuple(dudt.shape)} (this is what the MeanFlow target needs)")


if __name__ == "__main__":
    test_generator_shapes_and_grad()
    test_generator_jvp()
    print("\nAll U-Net generator tests passed.")
