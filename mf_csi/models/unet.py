"""U-Net generator for MeanFlow CSI prediction (the DiU backbone `f_G`).

Predicts the average-velocity field u given:
  * the noisy CSI frame  h  [B, 2, Nt, Nc],
  * the temporal latent  Z  [B, cond_channels, Nt, Nc]  (from the ConvLSTM encoder),
  * the MeanFlow time pair (r, t), each in [0, 1].

h and Z are concatenated along the channel axis to form the U-Net input. The
time pair is encoded with sinusoidal embeddings + per-variable MLPs that are then
summed (as in the MeanFlow paper) and injected into every residual block via
adaptive (FiLM-style) scale/shift conditioning.

Architecture (Appendix B):
    in_conv 3x3 -> 32ch
    enc1: 2x ResBlock(32)                              [skip1]            (Nt x Nc)
    down 2x
    enc2: 2x [ResBlock(->64) + SelfAttn]               [skip2]      (Nt/2 x Nc/2)
    mid : ResBlock - SelfAttn - ResBlock (64)                       (Nt/2 x Nc/2)
    up1 : cat(skip2); 3x [ResBlock(->64) + SelfAttn]               (Nt/2 x Nc/2)
    up  : 2x
    up2 : cat(skip1); 3x ResBlock(->32)                                 (Nt x Nc)
    head: GroupNorm -> SiLU -> Conv3x3 -> 2ch  (zero-init)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import UNetConfig


# --------------------------------------------------------------------------- #
# Time-pair embedding
# --------------------------------------------------------------------------- #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] -> [B, dim]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / (half - 1)
        )
        args = x[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimePairEmbedding(nn.Module):
    """Embed (r, t) -> [B, dim]: sinusoidal + 2-layer MLP per variable, summed."""

    def __init__(self, dim: int, time_scale: float = 1000.0):
        super().__init__()
        self.time_scale = time_scale
        self.sinu = SinusoidalPosEmb(dim)
        self.mlp_r = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.mlp_t = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        er = self.mlp_r(self.sinu(r * self.time_scale))
        et = self.mlp_t(self.sinu(t * self.time_scale))
        return er + et


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    """GroupNorm/SiLU/Conv residual block with FiLM conditioning from the time
    embedding (adaptive scale/shift after the second normalization)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int,
                 groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)     # -> (scale, shift)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Single-/multi-head self-attention over spatial positions."""

    def __init__(self, channels: int, num_heads: int = 1, groups: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]          # [B, heads, C/heads, HW]
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q, k) * scale, dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #
class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        in_total = cfg.in_channels + cfg.cond_channels
        c1 = cfg.base_channels
        c2 = cfg.base_channels * cfg.ch_mult
        emb, g, drop, heads = cfg.time_embed_dim, cfg.norm_groups, cfg.dropout, cfg.num_heads
        nres = cfg.num_res_blocks

        self.in_conv = nn.Conv2d(in_total, c1, 3, padding=1)

        # Encoder stage 1 (full res, c1)
        self.enc1 = nn.ModuleList([ResBlock(c1, c1, emb, g, drop) for _ in range(nres)])
        self.down = Downsample(c1)

        # Encoder stage 2 (half res, c2) with attention
        self.enc2, self.enc2_attn = nn.ModuleList(), nn.ModuleList()
        cin = c1
        for _ in range(nres):
            self.enc2.append(ResBlock(cin, c2, emb, g, drop))
            self.enc2_attn.append(SelfAttention2d(c2, heads, g))
            cin = c2

        # Bottleneck (half res, c2)
        self.mid_res1 = ResBlock(c2, c2, emb, g, drop)
        self.mid_attn = SelfAttention2d(c2, heads, g)
        self.mid_res2 = ResBlock(c2, c2, emb, g, drop)

        # Decoder up block 1 (half res): concat skip2 then 3x (ResBlock + Attn)
        self.up1, self.up1_attn = nn.ModuleList(), nn.ModuleList()
        cin = c2 + c2
        for _ in range(3):
            self.up1.append(ResBlock(cin, c2, emb, g, drop))
            self.up1_attn.append(SelfAttention2d(c2, heads, g))
            cin = c2
        self.up = Upsample(c2)

        # Decoder up block 2 (full res): concat skip1 then 3x ResBlock (no attn)
        self.up2 = nn.ModuleList()
        cin = c2 + c1
        for _ in range(3):
            self.up2.append(ResBlock(cin, c1, emb, g, drop))
            cin = c1

        # Output head (zero-init final conv: predict ~0 velocity at start)
        self.out_norm = nn.GroupNorm(g, c1)
        self.out_conv = nn.Conv2d(c1, cfg.out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        for blk in self.enc1:
            h = blk(h, emb)
        skip1 = h
        h = self.down(h)
        for blk, attn in zip(self.enc2, self.enc2_attn):
            h = attn(blk(h, emb))
        skip2 = h
        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)
        h = torch.cat([h, skip2], dim=1)
        for blk, attn in zip(self.up1, self.up1_attn):
            h = attn(blk(h, emb))
        h = self.up(h)
        h = torch.cat([h, skip1], dim=1)
        for blk in self.up2:
            h = blk(h, emb)
        return self.out_conv(F.silu(self.out_norm(h)))


class UNetGenerator(nn.Module):
    """f_G: predict average velocity u from (noisy CSI h, latent Z, r, t)."""

    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embed = TimePairEmbedding(cfg.time_embed_dim, cfg.time_scale)
        self.unet = UNet(cfg)

    def forward(self, h_noisy: torch.Tensor, z: torch.Tensor,
                r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(r, t)
        x = torch.cat([h_noisy, z], dim=1)
        return self.unet(x, emb)
