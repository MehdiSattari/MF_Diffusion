"""Diffusion core for DiU: DDIM schedule, training loss, and AR inference.

Matches the paper's original recipe: cosine (squaredcos_cap_v2) schedule with
T=2000, the network predicts the clean frame x0 (prediction_type='sample'),
Huber loss, per-sample random-SNR history corruption, and DDIM sampling that (by
default) starts the deterministic trajectory from ZEROS -- the paper's
"deterministic MMSE" behaviour -- with a growing history window in the AR rollout.
"""

from __future__ import annotations

from typing import Tuple, Dict
import torch
import torch.nn as nn

from .config import DiUConfig


def make_scheduler(cfg: DiUConfig):
    from diffusers import DDIMScheduler
    kw = dict(num_train_timesteps=cfg.num_train_timesteps,
              beta_schedule=cfg.beta_schedule,
              prediction_type=cfg.prediction_type)
    spacing = getattr(cfg, "timestep_spacing", "leading")
    try:
        return DDIMScheduler(**kw, timestep_spacing=spacing)
    except TypeError:                       # older diffusers: no timestep_spacing arg
        if spacing != "leading":
            print(f"[make_scheduler] diffusers too old for timestep_spacing={spacing}; using default")
        return DDIMScheduler(**kw)


def corrupt_history(past: torch.Tensor, snr_db_min: float, snr_db_max: float) -> torch.Tensor:
    """Additive noise at a per-sample random SNR (scale-preserving)."""
    B = past.shape[0]
    snr_db = torch.empty(B, device=past.device).uniform_(snr_db_min, snr_db_max)
    rho = 10.0 ** (snr_db / 10.0)
    power = past.pow(2).flatten(1).mean(dim=1)
    sigma = torch.sqrt(power / rho).view(B, *([1] * (past.dim() - 1)))
    return past + torch.randn_like(past) * sigma


def diffusion_loss(encoder, unet, scheduler, history, target, cfg: DiUConfig,
                   huber: nn.Module) -> Tuple[torch.Tensor, Dict]:
    """Single-next-frame diffusion loss (predict x0).

    history: [B, T, 2, Nt, Nc]  (conditioning window)
    target:  [B, 2, Nt, Nc]     (the clean next frame to predict)
    """
    device = target.device
    B = target.shape[0]

    hist = corrupt_history(history, cfg.train_snr_min, cfg.train_snr_max)
    z = encoder(hist)                                   # [B, z, Nt, Nc]

    x0 = target
    noise = torch.randn_like(x0)
    t = torch.randint(0, cfg.num_train_timesteps, (B,), device=device, dtype=torch.long)
    x_t = scheduler.add_noise(x0, noise, t)             # forward diffusion
    pred_x0 = unet(x_t, z, t)                           # predict clean frame
    loss = huber(pred_x0, x0)
    return loss, {"loss": loss.detach()}


@torch.no_grad()
def ddim_sample_next(unet, scheduler, z, cfg: DiUConfig, data_channels=2):
    """One next-frame DDIM sample conditioned on Z. Returns [B, 2, Nt, Nc]."""
    B, _, Nt, Nc = z.shape
    device = z.device
    scheduler.set_timesteps(cfg.sampling_steps, device=device)
    if cfg.deterministic_init:
        x = torch.zeros(B, data_channels, Nt, Nc, device=device)     # paper: start from zeros
    else:
        x = torch.randn(B, data_channels, Nt, Nc, device=device)
    for t in scheduler.timesteps:
        pred_x0 = unet(x, z, t)
        x = scheduler.step(pred_x0, t, x, eta=cfg.ddim_eta).prev_sample
    return x


@torch.no_grad()
def ddim_ar_predict(encoder, unet, scheduler, past, num_future, cfg: DiUConfig,
                    data_channels=2) -> torch.Tensor:
    """Autoregressive rollout with a growing history window. -> [B, num_future, 2, Nt, Nc]."""
    was = (encoder.training, unet.training)
    encoder.eval(); unet.eval()
    history = past
    preds = []
    for _ in range(num_future):
        z = encoder(history)
        nxt = ddim_sample_next(unet, scheduler, z, cfg, data_channels)
        preds.append(nxt)
        history = torch.cat([history, nxt.unsqueeze(1)], dim=1)      # window grows
    encoder.train(was[0]); unet.train(was[1])
    return torch.stack(preds, dim=1)
