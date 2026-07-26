"""Standalone autoregressive ConvLSTM baseline (the paper's ConvLSTM predictor).

ConvLSTM over the history -> next-frame point estimate. At inference the prediction is
appended to the history and the model re-run (AR rollout). This is the AR-inference
discriminative reference for the autoregressive comparisons (vs MeanFlow / DiU), as
distinct from the seq2seq JointRegressor. Trained with MSE; no output activation
because CSI is signed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ARConvLSTMConfig
from .conv_lstm import ConvLSTM


class ARConvLSTM(nn.Module):
    def __init__(self, cfg: ARConvLSTMConfig):
        super().__init__()
        self.convlstm = ConvLSTM(cfg.in_channels, cfg.hidden_channels,
                                 cfg.kernel_size, cfg.num_layers)
        self.norm = nn.GroupNorm(cfg.norm_groups, cfg.hidden_channels)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Conv2d(cfg.hidden_channels, cfg.in_channels, 3, padding=1)
        self.act = nn.Tanh() if cfg.final_activation == "tanh" else nn.Identity()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history [B, T, 2, Nt, Nc] -> next frame [B, 2, Nt, Nc]."""
        h, _ = self.convlstm(history)
        h = self.dropout(self.norm(h))
        return self.act(self.head(h))
