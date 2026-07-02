"""DiU model: ConvLSTM next-frame predictor + diffusers UNet2DModel.

Faithful to the paper's original code:
  * DiUEncoder: a ConvLSTM over the history whose final hidden state is mapped by a
    3x3 conv to a 2-channel next-frame estimate Z (ReLU). This is the conditioning.
  * DiUNet: a diffusers UNet2DModel with in_channels = 2 (noisy frame) + z_channels,
    out_channels = 2, a single resolution level of width 32, predicting the clean
    frame x0 (prediction_type='sample'). Z conditions the U-Net by channel-concat.

The diffusers UNet2DModel is used directly (same constructor args as the paper),
so the architecture matches exactly rather than being re-derived.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import DiUConfig
from .conv_lstm import ConvLSTM


class DiUEncoder(nn.Module):
    def __init__(self, cfg: DiUConfig, in_channels: int = 2):
        super().__init__()
        self.convlstm = ConvLSTM(in_channels, cfg.lstm_hidden,
                                 cfg.lstm_kernel, cfg.lstm_layers)
        self.conv_out = nn.Conv2d(cfg.lstm_hidden, cfg.z_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU() if cfg.lstm_activation == "relu" else nn.Identity()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history [B, T, C, Nt, Nc] -> Z [B, z_channels, Nt, Nc]."""
        h, _ = self.convlstm(history)          # final hidden state [B, hidden, Nt, Nc]
        return self.act(self.conv_out(h))


class DiUNet(nn.Module):
    def __init__(self, cfg: DiUConfig, data_channels: int = 2, image_size: int = 16):
        super().__init__()
        from diffusers import UNet2DModel      # imported lazily so the rest of the
        self.unet = UNet2DModel(               # package doesn't hard-require diffusers
            sample_size=image_size,
            in_channels=data_channels + cfg.z_channels,
            out_channels=data_channels,
            layers_per_block=cfg.unet_layers_per_block,
            block_out_channels=(cfg.unet_width,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=cfg.unet_norm_groups,
        )

    def forward(self, x_t: torch.Tensor, z: torch.Tensor, t) -> torch.Tensor:
        """x_t [B,2,H,W], z [B,z,H,W], t scalar or [B] -> predicted x0 [B,2,H,W]."""
        model_input = torch.cat([x_t, z], dim=1)
        return self.unet(model_input, t).sample
