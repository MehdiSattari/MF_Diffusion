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

    # The generator's final conv is zero-initialized (standard for flow models),
    # so at the very first step u==0 and gradient cannot reach anything upstream
    # of the output layer (encoder included). That is a step-0 artifact only.
    # Perturb the output weights so this test reflects the steady state where the
    # encoder does receive gradient.
    with torch.no_grad():
        gen.unet.out_conv.weight.normal_(0.0, 0.01)

    B = 2
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    past = torch.randn(B, Np, 2, Nt, Nc)
    future = torch.randn(B, Nf, 2, Nt, Nc)

    loss, metrics = meanflow_loss(enc, gen, past, future, cfg.meanflow)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    # Informative prior: auxiliary point-estimate loss must be reported and finite.
    assert "mu_mse" in metrics and torch.isfinite(metrics["mu_mse"]), metrics.get("mu_mse")
    loss.backward()

    # The mu head must receive gradient from the auxiliary MSE(mu, Y) term.
    mu_grads = [p.grad for p in enc.mu_head.parameters() if p.grad is not None]
    assert len(mu_grads) > 0 and all(torch.isfinite(g).all() for g in mu_grads), "mu head got no gradient"
    mu_norm = torch.sqrt(sum((g ** 2).sum() for g in mu_grads))
    assert mu_norm > 0, "mu-head gradient is zero — the point estimate would not train"

    # Gradients must reach BOTH sub-networks AND be non-zero. (The encoder
    # gradient being exactly zero was a real bug: computing u and d/dt u in one
    # dual pass drops the reverse edge to Z. This asserts the fix holds.)
    enc_grad = [p.grad for p in enc.parameters() if p.grad is not None]
    gen_grad = [p.grad for p in gen.parameters() if p.grad is not None]
    assert len(enc_grad) > 0, "no gradients reached the ENCODER"
    assert len(gen_grad) > 0, "no gradients reached the GENERATOR"
    assert all(torch.isfinite(g).all() for g in enc_grad + gen_grad)

    enc_norm = torch.sqrt(sum((g ** 2).sum() for g in enc_grad))
    gen_norm = torch.sqrt(sum((g ** 2).sum() for g in gen_grad))
    assert enc_norm > 0, "ENCODER gradient is zero — it would not train"
    assert gen_norm > 0, "GENERATOR gradient is zero — it would not train"
    print(f"OK meanflow loss={loss.item():.4f} mse={metrics['mse'].item():.4f} "
          f"mu_mse={metrics['mu_mse'].item():.4f} | "
          f"grad-norm enc={enc_norm:.3e} gen={gen_norm:.3e} mu={mu_norm:.3e}")


if __name__ == "__main__":
    test_sample_r_t()
    test_loss_and_param_grads()
    print("\nAll MeanFlow objective tests passed.")
