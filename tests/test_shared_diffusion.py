"""Shared-backbone diffusion: loss + AR sampling on the SAME UNetGenerator MeanFlow uses.

Verifies the controlled-comparison plumbing: the diffusion objective trains the shared
encoder + UNetGenerator, and DDIM AR sampling produces the right shape.

Run (needs torch + diffusers, no Sionna):  python -m tests.test_shared_diffusion
"""

import torch
import torch.nn as nn

from mf_csi.config import Config
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.diffusion_shared import (make_scheduler, diffusion_loss_shared,
                                     ddim_ar_predict_shared)


def test_shared_diffusion_loss_grad_and_sampling():
    cfg = Config()
    enc = TemporalEncoder(cfg.encoder)
    gen = UNetGenerator(cfg.generator)
    scheduler = make_scheduler(cfg.diu)
    huber = nn.HuberLoss(delta=cfg.diu.huber_delta)

    # Zero-init output conv blocks gradient upstream at step 0; perturb so the test
    # reflects steady-state gradient flow to BOTH encoder and generator.
    with torch.no_grad():
        gen.unet.out_conv.weight.normal_(0.0, 0.01)

    B = 2
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    past = torch.randn(B, Np, 2, Nt, Nc)
    future = torch.randn(B, Nf, 2, Nt, Nc)

    loss, _ = diffusion_loss_shared(enc, gen, scheduler, past, future[:, 0], cfg.diu, huber)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    loss.backward()
    enc_g = [p.grad for p in enc.parameters() if p.grad is not None]
    gen_g = [p.grad for p in gen.parameters() if p.grad is not None]
    assert len(enc_g) > 0 and len(gen_g) > 0, "gradient missing to encoder or generator"
    assert all(torch.isfinite(g).all() for g in enc_g + gen_g)

    # Short AR rollout (few steps) -> correct horizon shape.
    cfg.diu.sampling_steps = 3
    pred = ddim_ar_predict_shared(enc, gen, scheduler, past, Nf, cfg.diu)
    assert pred.shape == (B, Nf, 2, Nt, Nc), pred.shape
    print(f"OK shared diffusion | loss {loss.item():.4f} | AR -> {tuple(pred.shape)}")


if __name__ == "__main__":
    test_shared_diffusion_loss_grad_and_sampling()
    print("\nShared-backbone diffusion test passed.")
