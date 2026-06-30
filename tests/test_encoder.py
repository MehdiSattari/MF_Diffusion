"""Shape + gradient tests for the ConvLSTM temporal encoder.

Run (needs torch):  python -m tests.test_encoder
"""

import torch

from mf_csi.config import Config
from mf_csi.models import TemporalEncoder, ConvLSTM


def test_convlstm_final_hidden_shape():
    B, T, C, H, W = 2, 5, 2, 16, 16
    net = ConvLSTM(in_channels=C, hidden_channels=32, kernel_size=3, num_layers=2)
    x = torch.randn(B, T, C, H, W)
    hidden, states = net(x)
    assert hidden.shape == (B, 32, H, W), hidden.shape
    assert len(states) == 2
    print("OK ConvLSTM hidden", tuple(hidden.shape))


def test_encoder_shapes_and_grad():
    cfg = Config()
    enc = TemporalEncoder(cfg.encoder)
    B = 4
    Np, Nt, Nc = cfg.data.num_past, cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    x = torch.randn(B, Np, cfg.encoder.in_channels, Nt, Nc, requires_grad=True)
    z = enc(x)
    assert z.shape == (B, cfg.encoder.latent_channels, Nt, Nc), z.shape
    # Gradient should flow back to the input.
    z.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"OK TemporalEncoder Z {tuple(z.shape)} | params {n_params/1e3:.1f}k")


if __name__ == "__main__":
    test_convlstm_final_hidden_shape()
    test_encoder_shapes_and_grad()
    print("\nAll encoder tests passed.")
