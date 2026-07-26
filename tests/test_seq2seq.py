"""Encoder-free seq2seq core: MeanFlow + diffusion on one shared 2D U-Net.

Verifies frame<->channel roundtrip, that both objectives train the shared backbone
(gradients flow), and that both produce the full future block. Also checks the
small (no-attention) backbone builds.

Run (needs torch + diffusers, no Sionna):  python -m tests.test_seq2seq
"""

import torch
import torch.nn as nn

from mf_csi.config import Config
from mf_csi.models import UNetGenerator
from mf_csi.seq2seq import (frames_to_channels, channels_to_frames, seq2seq_unet_config,
                            meanflow_seq2seq_loss, meanflow_seq2seq_predict,
                            diffusion_seq2seq_loss, diffusion_seq2seq_predict,
                            make_scheduler)


def test_reshape_roundtrip():
    x = torch.randn(2, 10, 2, 16, 16)
    y = channels_to_frames(frames_to_channels(x), 10)
    assert torch.allclose(x, y), "frame<->channel roundtrip broken"
    print("OK reshape roundtrip")


def _build(cfg, size):
    ucfg = seq2seq_unet_config(cfg.data.num_past, cfg.data.num_future,
                               ch_mult=(2 if size == "large" else 1),
                               num_res_blocks=(2 if size == "large" else 1),
                               use_attention=(size == "large"))
    return UNetGenerator(ucfg)


def test_meanflow_and_diffusion_seq2seq():
    cfg = Config()
    B = 2
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    past = torch.randn(B, Np, 2, Nt, Nc)
    future = torch.randn(B, Nf, 2, Nt, Nc)

    for size in ("small", "large"):
        gen = _build(cfg, size)
        with torch.no_grad():                       # unblock zero-init output conv
            gen.unet.out_conv.weight.normal_(0.0, 0.01)

        # MeanFlow
        loss, _ = meanflow_seq2seq_loss(gen, past, future, cfg.meanflow)
        assert loss.ndim == 0 and torch.isfinite(loss), loss
        loss.backward()
        g = [p.grad for p in gen.parameters() if p.grad is not None]
        assert len(g) > 0 and all(torch.isfinite(t).all() for t in g), "no MeanFlow grad"
        gen.zero_grad()

        # Diffusion
        scheduler = make_scheduler(cfg.diu)
        huber = nn.HuberLoss(delta=cfg.diu.huber_delta)
        dloss, _ = diffusion_seq2seq_loss(gen, scheduler, past, future, cfg.diu, huber)
        assert dloss.ndim == 0 and torch.isfinite(dloss), dloss
        dloss.backward()
        assert any(p.grad is not None for p in gen.parameters()), "no diffusion grad"

        # Prediction shapes
        pm = meanflow_seq2seq_predict(gen, past, Nf, seed_std=cfg.meanflow.source_std)
        cfg.diu.sampling_steps = 3
        pd = diffusion_seq2seq_predict(gen, scheduler, past, Nf, cfg.diu)
        assert pm.shape == (B, Nf, 2, Nt, Nc) and pd.shape == (B, Nf, 2, Nt, Nc)
        n = sum(p.numel() for p in gen.parameters())
        print(f"OK seq2seq [{size:5s}] | mf_loss {loss.item():.4f} diff_loss {dloss.item():.4f} "
              f"| gen params {n/1e6:.3f}M | pred {tuple(pm.shape)}")


if __name__ == "__main__":
    test_reshape_roundtrip()
    test_meanflow_and_diffusion_seq2seq()
    print("\nSeq2seq core tests passed.")
