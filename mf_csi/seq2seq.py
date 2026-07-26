"""Encoder-free SEQ2SEQ CSI prediction: MeanFlow vs diffusion on one shared backbone.

No temporal encoder. The history H_p is stacked into the channel axis and used as
conditioning; the whole future block H_f (all Nf frames) is predicted in a SINGLE
pass. Both generative models call the same `UNetGenerator`, so the ONLY difference
is the objective + sampling:

    frames_to_channels:  [B, T, 2, Nt, Nc]  ->  [B, T*2, Nt, Nc]
    generator input   :  concat(noisy_future_block [Nf*2], past_block [Np*2])
    generator output  :  future_block [Nf*2]

    MeanFlow : predict average velocity of the block, 1-NFE
    Diffusion: predict the clean block x0, multi-step DDIM
    ConvLSTM : JointRegressor (separate module) -- the discriminative seq2seq baseline

Prior: plain N(0, sigma^2) (no learned point estimate -> no mu asymmetry). With
global-std normalization the data is ~unit variance, so sigma=1 is well scaled.
A symmetric persistence init (last observed frame) can be added later if 1-NFE
MeanFlow needs an anchor; it would be applied to BOTH models identically.
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

from .config import MeanFlowConfig, DiUConfig
from .meanflow import sample_r_t, augment_history
from .diffusion import make_scheduler, corrupt_history


# --------------------------------------------------------------------------- #
# frame <-> channel reshaping
# --------------------------------------------------------------------------- #
def frames_to_channels(x: torch.Tensor) -> torch.Tensor:
    """[B, T, 2, Nt, Nc] -> [B, T*2, Nt, Nc]."""
    B, T, C, Nt, Nc = x.shape
    return x.reshape(B, T * C, Nt, Nc)


def channels_to_frames(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    """[B, F*2, Nt, Nc] -> [B, F, 2, Nt, Nc]."""
    B, FC, Nt, Nc = x.shape
    return x.reshape(B, num_frames, FC // num_frames, Nt, Nc)


# --------------------------------------------------------------------------- #
# MeanFlow (seq2seq, 1-NFE)
# --------------------------------------------------------------------------- #
def meanflow_seq2seq_loss(generator, past: torch.Tensor, future: torch.Tensor,
                          cfg: MeanFlowConfig) -> Tuple[torch.Tensor, Dict]:
    """MeanFlow loss on the whole future block. past/future: [B, T, 2, Nt, Nc]."""
    device = future.device
    B, Nf = future.shape[0], future.shape[1]
    cond = frames_to_channels(augment_history(past, cfg))    # [B, Np*2, Nt, Nc]
    Y = frames_to_channels(future)                           # [B, Nf*2, Nt, Nc]
    eps = torch.randn_like(Y) * cfg.source_std               # plain source (no mu)
    r, t = sample_r_t(B, cfg, device)
    t_b = t.view(B, 1, 1, 1)
    Ht = (1.0 - t_b) * Y + t_b * eps
    v = eps - Y

    u = generator(Ht, cond, r, t)                            # [B, Nf*2, Nt, Nc]
    with fwAD.dual_level():
        h_dual = fwAD.make_dual(Ht.detach(), v)
        r_dual = fwAD.make_dual(r.detach(), torch.zeros_like(r))
        t_dual = fwAD.make_dual(t.detach(), torch.ones_like(t))
        u_dual = generator(h_dual, cond.detach(), r_dual, t_dual)
        dudt = fwAD.unpack_dual(u_dual).tangent
    dudt = (torch.zeros_like(u) if dudt is None else dudt).detach()

    tr = (t - r).view(B, 1, 1, 1)
    u_tgt = (v - tr * dudt).detach()
    err = u - u_tgt
    sq = err.pow(2).flatten(1).mean(dim=1)
    w = 1.0 / (sq.detach() + cfg.loss_eps).pow(cfg.loss_power)
    loss = (w * sq).mean()
    return loss, {"loss": loss.detach(), "mse": sq.mean().detach(),
                  "frac_r_neq_t": (r != t).float().mean().detach()}


@torch.no_grad()
def meanflow_seq2seq_predict(generator, past: torch.Tensor, num_future: int,
                             seed_std: float = 1.0, num_samples: int = 1) -> torch.Tensor:
    """1-NFE seq2seq prediction -> [B, num_future, 2, Nt, Nc]."""
    was = generator.training
    generator.eval()
    B, _, _, Nt, Nc = past.shape
    device = past.device
    cond = frames_to_channels(past)
    out_ch = num_future * 2
    r = torch.zeros(B, device=device); t = torch.ones(B, device=device)
    acc = torch.zeros(B, out_ch, Nt, Nc, device=device)
    for _ in range(max(1, num_samples)):
        S = torch.randn(B, out_ch, Nt, Nc, device=device) * seed_std
        acc = acc + (S - generator(S, cond, r, t))
    yhat = acc / max(1, num_samples)
    generator.train(was)
    return channels_to_frames(yhat, num_future)


# --------------------------------------------------------------------------- #
# Diffusion (seq2seq, multi-step DDIM over the whole block)
# --------------------------------------------------------------------------- #
def diffusion_seq2seq_loss(generator, scheduler, past: torch.Tensor, future: torch.Tensor,
                           cfg: DiUConfig, huber: nn.Module,
                           snr_min: float = -20.0, snr_max: float = 20.0
                           ) -> Tuple[torch.Tensor, Dict]:
    """Predict-x0 diffusion loss on the whole future block."""
    device = future.device
    B = future.shape[0]
    cond = frames_to_channels(corrupt_history(past, snr_min, snr_max))
    x0 = frames_to_channels(future)
    noise = torch.randn_like(x0)
    t = torch.randint(0, cfg.num_train_timesteps, (B,), device=device, dtype=torch.long)
    x_t = scheduler.add_noise(x0, noise, t)
    tn = t.float() / float(cfg.num_train_timesteps)
    pred_x0 = generator(x_t, cond, tn, tn)
    loss = huber(pred_x0, x0)
    return loss, {"loss": loss.detach()}


@torch.no_grad()
def diffusion_seq2seq_predict(generator, scheduler, past: torch.Tensor, num_future: int,
                              cfg: DiUConfig) -> torch.Tensor:
    """Multi-step DDIM prediction of the whole block -> [B, num_future, 2, Nt, Nc]."""
    was = generator.training
    generator.eval()
    B, _, _, Nt, Nc = past.shape
    device = past.device
    cond = frames_to_channels(past)
    out_ch = num_future * 2
    scheduler.set_timesteps(cfg.sampling_steps, device=device)
    x = (torch.zeros if cfg.deterministic_init else torch.randn)(B, out_ch, Nt, Nc, device=device)
    for t in scheduler.timesteps:
        tn = (t.float() / float(cfg.num_train_timesteps)).expand(B).to(device)
        pred_x0 = generator(x, cond, tn, tn)
        x = scheduler.step(pred_x0, t, x, eta=cfg.ddim_eta).prev_sample
    generator.train(was)
    return channels_to_frames(x, num_future)


def seq2seq_unet_config(num_past: int, num_future: int, base_channels: int = 32,
                        ch_mult: int = 2, num_res_blocks: int = 2,
                        use_attention: bool = True):
    """Build a UNetConfig for the encoder-free seq2seq backbone: noisy future block
    (Nf*2 ch) conditioned on the past block (Np*2 ch), predicting the future block."""
    from .config import UNetConfig
    return UNetConfig(
        in_channels=num_future * 2,
        cond_channels=num_past * 2,
        out_channels=num_future * 2,
        base_channels=base_channels,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        use_attention=use_attention,
    )
