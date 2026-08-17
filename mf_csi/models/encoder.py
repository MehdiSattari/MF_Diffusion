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

Informative-prior extension: when ``cfg.predict_mu`` is set, a second lightweight
head maps the SAME ConvLSTM hidden state to a 2-channel next-frame point estimate
``mu(Z) ~= E[Y | history]``. This mu is used to CENTER the MeanFlow source
distribution (H^1 ~ N(mu, sigma^2)) so a single 1-NFE draw already sits near the
conditional mean -- recovering NMSE without narrowing the prior. mu is trained by
an auxiliary MSE-to-Y loss (see meanflow.py); the conditioning latent Z is
unchanged, so nothing about the generator's interface changes.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

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

        # Optional next-frame point-estimate head (linear output: std-normalized
        # CSI is signed, so no activation). Reads the ConvLSTM hidden state directly.
        self.predict_mu = bool(getattr(cfg, "predict_mu", False))
        if self.predict_mu:
            self.mu_head = nn.Conv2d(cfg.hidden_channels,
                                     getattr(cfg, "mu_channels", 2),
                                     kernel_size=3, padding=1)

    def _heads(self, hidden: torch.Tensor, return_mu: bool):
        z = self.final_act(self.proj(self.dropout(self.norm(hidden))))
        if not return_mu:
            return z
        mu = self.mu_head(hidden) if self.predict_mu else None
        return z, mu

    def forward(self, h_past: torch.Tensor, return_mu: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """h_past: [B, Np, in_channels, Nt, Nc].

        return_mu=False -> Z [B, latent_channels, Nt, Nc]  (backward compatible).
        return_mu=True  -> (Z, mu) with mu [B, mu_channels, Nt, Nc] or None.
        """
        hidden, _ = self.convlstm(h_past)     # [B, hidden, Nt, Nc]
        return self._heads(hidden, return_mu)

    def warmup(self, h_past: torch.Tensor):
        """Process the history ONCE to obtain the recurrent state (O(Np)). Returns
        (states, top_hidden); use step() thereafter for O(1)-per-frame AR inference."""
        top_hidden, states = self.convlstm(h_past)
        return states, top_hidden

    def step(self, frame: torch.Tensor, states, return_mu: bool = False):
        """Advance the ConvLSTM by ONE frame (O(1)) and emit (z[, mu]) + new states.
        frame: [B, in_channels, Nt, Nc]. Equivalent to re-encoding the full history
        but at constant per-step cost."""
        top_hidden, new_states = self.convlstm.step(frame, states)
        return self._heads(top_hidden, return_mu), new_states
