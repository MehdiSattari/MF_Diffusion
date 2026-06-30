"""DiU temporal encoder: ConvLSTM over the CSI history -> conditioning latent Z.

Pipeline (matches the paper's ConvLSTM description):
    H_past [B, Np, 2, Nt, Nc]
      -> ConvLSTM            -> top hidden state [B, hidden, Nt, Nc]
      -> GroupNorm(1 group)
      -> Dropout(p)
      -> Conv2d 3x3          -> Z [B, latent_channels, Nt, Nc]
      -> (optional) tanh

Z is a spatial feature map so it can be concatenated channel-wise with the noisy
CSI frame inside the U-Net generator (Step 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import EncoderConfig
from .conv_lstm import ConvLSTM


class TemporalEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.convlstm = ConvLSTM(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            kernel_size=cfg.kernel_size,
            num_layers=cfg.num_layers,
        )
        self.norm = nn.GroupNorm(cfg.norm_groups, cfg.hidden_channels)
        self.dropout = nn.Dropout(cfg.dropout)
        self.proj = nn.Conv2d(cfg.hidden_channels, cfg.latent_channels,
                              kernel_size=3, padding=1)
        if cfg.final_activation == "tanh":
            self.final_act: nn.Module = nn.Tanh()
        elif cfg.final_activation == "none":
            self.final_act = nn.Identity()
        else:
            raise ValueError(f"Unknown final_activation: {cfg.final_activation}")

    def forward(self, h_past: torch.Tensor) -> torch.Tensor:
        """h_past: [B, Np, in_channels, Nt, Nc] -> Z: [B, latent_channels, Nt, Nc]."""
        hidden, _ = self.convlstm(h_past)     # [B, hidden, Nt, Nc]
        z = self.norm(hidden)
        z = self.dropout(z)
        z = self.proj(z)
        z = self.final_act(z)
        return z
