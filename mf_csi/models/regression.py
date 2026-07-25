"""Joint-horizon ConvLSTM regression baseline (JointRegressor).

A ConvLSTM temporal encoder over the history feeds a small convolutional decoder
that emits ALL Nf future frames in a SINGLE forward pass. There is no
autoregressive rollout, so there is no exposure bias -- this is the rollout-free
regression 'ceiling' and a clean ConvLSTM reference against which the diffusion
(DiU) and MeanFlow generators are compared. Trained with plain MSE.

    H_past [B, Np, 2, Nt, Nc]
      -> ConvLSTM (hidden C)         -> top hidden [B, C, Nt, Nc]
      -> Conv 3x3 -> C               -> ResBlock x num_res_blocks
      -> GroupNorm -> SiLU -> Conv   -> [B, Nf*2, Nt, Nc]
      -> reshape                     -> [B, Nf, 2, Nt, Nc]   (all frames at once)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RegressionConfig
from .conv_lstm import ConvLSTM


class ResBlock(nn.Module):
    """GroupNorm/SiLU/Conv residual block (no time conditioning -- pure regressor)."""

    def __init__(self, ch: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class JointRegressor(nn.Module):
    """ConvLSTM encoder + conv decoder emitting all Nf frames in one forward pass."""

    def __init__(self, cfg: RegressionConfig):
        super().__init__()
        self.cfg = cfg
        self.num_future = cfg.num_future
        self.convlstm = ConvLSTM(cfg.in_channels, cfg.hidden_channels,
                                 cfg.kernel_size, cfg.num_layers)
        self.in_proj = nn.Conv2d(cfg.hidden_channels, cfg.decoder_channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            ResBlock(cfg.decoder_channels, cfg.norm_groups, cfg.dropout)
            for _ in range(cfg.num_res_blocks)
        ])
        self.out_norm = nn.GroupNorm(cfg.norm_groups, cfg.decoder_channels)
        self.out_conv = nn.Conv2d(cfg.decoder_channels, cfg.num_future * 2, 3, padding=1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history [B, Np, 2, Nt, Nc] -> preds [B, Nf, 2, Nt, Nc]."""
        B = history.shape[0]
        h, _ = self.convlstm(history)                 # [B, hidden, Nt, Nc]
        h = self.in_proj(h)
        for blk in self.blocks:
            h = blk(h)
        out = self.out_conv(F.silu(self.out_norm(h)))  # [B, Nf*2, Nt, Nc]
        Nt, Nc = out.shape[-2], out.shape[-1]
        return out.view(B, self.num_future, 2, Nt, Nc)
