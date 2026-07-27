"""2x2 mu-ablation paths on the shared backbone: meanflow +/-mu and diffusion +/-mu.

Verifies both objectives train the shared encoder+UNetGenerator with and without the
informative mu prior, that MeanFlow's forward-AD JVP still works, that the residual-mu
diffusion loss is finite, and that AR sampling yields the right shape.

Run (needs torch + diffusers):  python -m tests.test_ar_ablation
"""

import torch
import torch.nn as nn
from dataclasses import replace

from mf_csi.config import Config
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.meanflow import meanflow_loss
from mf_csi.diffusion_shared import make_scheduler, diffusion_loss_shared, ddim_ar_predict_shared
from mf_csi.inference import autoregressive_predict


def _data(cfg, B=2):
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    return (torch.randn(B, Np, 2, Nt, Nc), torch.randn(B, Nf, 2, Nt, Nc))


def test_meanflow_mu_ablation():
    cfg = Config()
    past, future = _data(cfg)
    for mu_on in (True, False):
        enc = TemporalEncoder(cfg.encoder); gen = UNetGenerator(cfg.generator)
        with torch.no_grad():
            gen.unet.out_conv.weight.normal_(0.0, 0.01)   # unblock zero-init for grad/JVP
        mfcfg = replace(cfg.meanflow, informative_prior=mu_on)
        loss, m = meanflow_loss(enc, gen, past, future, mfcfg)   # exercises the JVP
        assert loss.ndim == 0 and torch.isfinite(loss), loss
        loss.backward()
        assert any(p.grad is not None for p in enc.parameters())
        assert any(p.grad is not None for p in gen.parameters())
        print(f"OK meanflow mu={'on' if mu_on else 'off'} | loss {loss.item():.4f} mu_mse {m['mu_mse'].item():.4f}")


def test_diffusion_mu_ablation():
    cfg = Config(); cfg.diu.sampling_steps = 3
    past, future = _data(cfg)
    scheduler = make_scheduler(cfg.diu); huber = nn.HuberLoss(delta=cfg.diu.huber_delta)
    Nf = cfg.data.num_future
    for mu_on in (True, False):
        enc = TemporalEncoder(cfg.encoder); gen = UNetGenerator(cfg.generator)
        loss, m = diffusion_loss_shared(enc, gen, scheduler, past, future[:, 0], cfg.diu, huber, use_mu=mu_on)
        assert loss.ndim == 0 and torch.isfinite(loss), loss
        loss.backward()
        assert any(p.grad is not None for p in enc.parameters())
        pred = ddim_ar_predict_shared(enc, gen, scheduler, past, Nf, cfg.diu, use_mu=mu_on)
        assert pred.shape == (past.shape[0], Nf, 2, cfg.data.num_bs_ant, cfg.data.num_subcarriers_used)
        print(f"OK diffusion mu={'on' if mu_on else 'off'} | loss {loss.item():.4f} aux {m['aux'].item():.4f} "
              f"| AR {tuple(pred.shape)}")


if __name__ == "__main__":
    test_meanflow_mu_ablation()
    test_diffusion_mu_ablation()
    print("\n2x2 AR mu-ablation tests passed.")
