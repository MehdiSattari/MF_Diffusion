"""Diffusion objective on the SHARED UNetGenerator backbone.

For the controlled MeanFlow-vs-diffusion comparison, DiU and MeanFlow must use the
*same* temporal encoder and the *same* generator backbone, differing only in the
training objective and sampling. This module implements the diffusion path on the
shared `UNetGenerator` (the exact network MeanFlow uses), so the only variables are:

    MeanFlow : predict average velocity u(h, z, r, t), 1-NFE sampling
    Diffusion: predict clean x0 from a noised frame, multi-step DDIM sampling

Both call `UNetGenerator(h, z, r, t)`; diffusion simply passes the (normalized)
diffusion timestep as r = t. Everything else -- encoder, backbone weights budget,
normalization, optimizer, EMA, data, AR inference -- is identical.
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn

from .config import DiUConfig
from .diffusion import make_scheduler, corrupt_history


def _tn(t: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """Normalize an integer diffusion timestep to [0, 1] for the time embedding."""
    return t.float() / float(num_train_timesteps)


def diffusion_loss_shared(encoder, generator, scheduler, history, target,
                          cfg: DiUConfig, huber: nn.Module,
                          snr_min: float = -20.0, snr_max: float = 20.0
                          ) -> Tuple[torch.Tensor, Dict]:
    """Predict-x0 diffusion loss on the shared UNetGenerator.

    history: [B, T, 2, Nt, Nc] conditioning window; target: [B, 2, Nt, Nc] clean next frame.
    """
    device = target.device
    B = target.shape[0]
    hist = corrupt_history(history, snr_min, snr_max)
    z = encoder(hist)                                        # [B, Cz, Nt, Nc]
    x0 = target
    noise = torch.randn_like(x0)
    t = torch.randint(0, cfg.num_train_timesteps, (B,), device=device, dtype=torch.long)
    x_t = scheduler.add_noise(x0, noise, t)
    tn = _tn(t, cfg.num_train_timesteps)                     # [B] in [0,1]
    pred_x0 = generator(x_t, z, tn, tn)                      # r = t (diffusion has one time)
    loss = huber(pred_x0, x0)
    return loss, {"loss": loss.detach()}


@torch.no_grad()
def ddim_sample_next_shared(encoder, generator, scheduler, z, cfg: DiUConfig,
                            data_channels: int = 2):
    B, _, Nt, Nc = z.shape
    device = z.device
    scheduler.set_timesteps(cfg.sampling_steps, device=device)
    if cfg.deterministic_init:
        x = torch.zeros(B, data_channels, Nt, Nc, device=device)
    else:
        x = torch.randn(B, data_channels, Nt, Nc, device=device)
    for t in scheduler.timesteps:
        tn = _tn(t, cfg.num_train_timesteps).expand(B).to(device)
        pred_x0 = generator(x, z, tn, tn)
        x = scheduler.step(pred_x0, t, x, eta=cfg.ddim_eta).prev_sample
    return x


@torch.no_grad()
def ddim_ar_predict_shared(encoder, generator, scheduler, past, num_future,
                           cfg: DiUConfig, data_channels: int = 2) -> torch.Tensor:
    """Autoregressive DDIM rollout on the shared backbone -> [B, num_future, 2, Nt, Nc]."""
    was = (encoder.training, generator.training)
    encoder.eval(); generator.eval()
    history = past
    preds = []
    for _ in range(num_future):
        z = encoder(history)
        nxt = ddim_sample_next_shared(encoder, generator, scheduler, z, cfg, data_channels)
        preds.append(nxt)
        history = torch.cat([history, nxt.unsqueeze(1)], dim=1)
    encoder.train(was[0]); generator.train(was[1])
    return torch.stack(preds, dim=1)
