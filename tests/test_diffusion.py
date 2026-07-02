"""Shape / gradient / sampling tests for the diffusion DiU.

Needs diffusers (and torch). Run:  python -m tests.test_diffusion
"""

import torch
import torch.nn as nn

from mf_csi.config import Config
from mf_csi.models.diu import DiUEncoder, DiUNet
from mf_csi.diffusion import make_scheduler, diffusion_loss, ddim_ar_predict


def _build(cfg, device="cpu"):
    enc = DiUEncoder(cfg.diu, in_channels=2).to(device)
    unet = DiUNet(cfg.diu, data_channels=2, image_size=cfg.data.num_subcarriers_used).to(device)
    return enc, unet


def test_encoder_and_unet_shapes():
    cfg = Config()
    enc, unet = _build(cfg)
    B, Nt, Nc = 2, cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    hist = torch.randn(B, 7, 2, Nt, Nc)
    z = enc(hist)
    assert z.shape == (B, cfg.diu.z_channels, Nt, Nc), z.shape
    x_t = torch.randn(B, 2, Nt, Nc)
    t = torch.randint(0, cfg.diu.num_train_timesteps, (B,))
    out = unet(x_t, z, t)
    assert out.shape == (B, 2, Nt, Nc), out.shape
    print(f"OK shapes | Z {tuple(z.shape)} unet-out {tuple(out.shape)}")


def test_loss_and_grads():
    cfg = Config()
    enc, unet = _build(cfg)
    scheduler = make_scheduler(cfg.diu)
    huber = nn.HuberLoss(delta=cfg.diu.huber_delta)
    B, Nt, Nc = 2, cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    history = torch.randn(B, 12, 2, Nt, Nc)
    target = torch.randn(B, 2, Nt, Nc)
    loss, m = diffusion_loss(enc, unet, scheduler, history, target, cfg.diu, huber)
    loss.backward()
    enc_g = sum(p.grad.abs().sum() for p in enc.parameters() if p.grad is not None)
    unet_g = sum(p.grad.abs().sum() for p in unet.parameters() if p.grad is not None)
    assert enc_g > 0 and unet_g > 0, (enc_g, unet_g)
    n = sum(p.numel() for p in list(enc.parameters()) + list(unet.parameters()))
    print(f"OK loss={loss.item():.4f} | grads enc>0 unet>0 | params {n/1e6:.2f}M")


def test_ar_sampling_shape():
    cfg = Config()
    cfg.diu.sampling_steps = 4                    # keep the test fast
    enc, unet = _build(cfg)
    scheduler = make_scheduler(cfg.diu)
    B, Np, Nt, Nc = 2, cfg.data.num_past, cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    past = torch.randn(B, Np, 2, Nt, Nc)
    pred = ddim_ar_predict(enc, unet, scheduler, past, cfg.data.num_future, cfg.diu)
    assert pred.shape == (B, cfg.data.num_future, 2, Nt, Nc), pred.shape
    print(f"OK DDIM AR rollout -> {tuple(pred.shape)}")


if __name__ == "__main__":
    test_encoder_and_unet_shapes()
    test_loss_and_grads()
    test_ar_sampling_shape()
    print("\nAll diffusion-DiU tests passed.")
