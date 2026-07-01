"""Tests for the MeanFlow objective.

The critical check is that the loss backpropagates into BOTH the encoder and the
generator weights — this validates that forward-mode AD (used for d/dt u)
composes with reverse-mode autograd. If gradients were missing, the model would
silently fail to train.

Run (needs torch):  python -m tests.test_meanflow
"""

import torch

from mf_csi.config import Config
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.meanflow import meanflow_loss, sample_r_t


def test_sample_r_t():
    cfg = Config()
    r, t = sample_r_t(4096, cfg.meanflow, "cpu")
    assert (t >= r).all(), "t must be >= r"
    assert r.min() >= 0 and t.max() <= 1
    frac_neq = (r != t).float().mean().item()
    # Should be roughly ratio_r_not_equal_t (0.25) within sampling noise.
    assert 0.15 < frac_neq < 0.35, frac_neq
    print(f"OK sample_r_t | frac(r!=t)={frac_neq:.3f} (target {cfg.meanflow.ratio_r_not_equal_t})")


def test_loss_and_param_grads():
    cfg = Config()
    enc = TemporalEncoder(cfg.encoder)
    gen = UNetGenerator(cfg.generator)

    B = 2
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    past = torch.randn(B, Np, 2, Nt, Nc)
    future = torch.randn(B, Nf, 2, Nt, Nc)

    loss, metrics = meanflow_loss(enc, gen, past, future, cfg.meanflow)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    loss.backward()

    # Gradients must reach BOTH sub-networks (this is the forward/reverse-AD check).
    enc_grad = [p.grad for p in enc.parameters() if p.grad is not None]
    gen_grad = [p.grad for p in gen.parameters() if p.grad is not None]
    assert len(enc_grad) > 0, "no gradients reached the ENCODER"
    assert len(gen_grad) > 0, "no gradients reached the GENERATOR"
    assert all(torch.isfinite(g).all() for g in enc_grad + gen_grad)

    enc_norm = torch.sqrt(sum((g ** 2).sum() for g in enc_grad))
    gen_norm = torch.sqrt(sum((g ** 2).sum() for g in gen_grad))
    print(f"OK meanflow loss={loss.item():.4f} mse={metrics['mse'].item():.4f} | "
          f"grad-norm enc={enc_norm:.3e} gen={gen_norm:.3e}")


if __name__ == "__main__":
    test_sample_r_t()
    test_loss_and_param_grads()
    print("\nAll MeanFlow objective tests passed.")
