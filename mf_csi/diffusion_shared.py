"""Diffusion objective on the SHARED UNetGenerator backbone, with an optional
informative (mu) prior -- the symmetric analogue of MeanFlow's informative prior.

For the controlled 2x2 mu-ablation, DiU and MeanFlow use the SAME temporal encoder
and the SAME UNetGenerator; only the objective + sampling + the mu switch differ.

  use_mu = False : standard predict-x0 diffusion (source ~ N(0, I)).
  use_mu = True  : RESIDUAL diffusion -- the network models (x0 - mu) around the
                   encoder's point estimate mu (residual-diffusion / ResShift-style),
                   with an auxiliary MSE(mu, Y). At sampling, x0 = mu + residual.
                   This mirrors MeanFlow centering its source on mu, so the mu switch
                   means the same thing for both objectives.

Diffusion passes the (normalized) timestep as r = t to the shared UNetGenerator.
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiUConfig
from .diffusion import make_scheduler, corrupt_history


def _tn(t: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    return t.float() / float(num_train_timesteps)


def diffusion_loss_shared(encoder, generator, scheduler, history, target,
                          cfg: DiUConfig, huber: nn.Module,
                          snr_min: float = -20.0, snr_max: float = 20.0,
                          use_mu: bool = False, mu_weight: float = 1.0
                          ) -> Tuple[torch.Tensor, Dict]:
    """Predict-x0 (or predict-residual, if use_mu) diffusion loss on the shared U-Net.

    history: [B, T, 2, Nt, Nc]; target: [B, 2, Nt, Nc] clean next frame.
    """
    device = target.device
    B = target.shape[0]
    hist = corrupt_history(history, snr_min, snr_max)
    z, mu = encoder(hist, return_mu=True)
    x0 = target
    if use_mu and mu is not None:
        base = mu.detach()
        diff_target = x0 - base                       # model the residual around mu
    else:
        diff_target = x0
    noise = torch.randn_like(diff_target)
    t = torch.randint(0, cfg.num_train_timesteps, (B,), device=device, dtype=torch.long)
    x_t = scheduler.add_noise(diff_target, noise, t)
    tn = _tn(t, cfg.num_train_timesteps)
    pred = generator(x_t, z, tn, tn)                  # network output (interpreted per prediction_type)
    # Regression target depends on the parameterization (must match the scheduler's
    # prediction_type so multi-step DDIM sampling is consistent):
    #   'sample'        -> predict the clean signal (residual or x0)  [default]
    #   'epsilon'       -> predict the added noise
    #   'v_prediction'  -> predict the velocity v = sqrt(abar) eps - sqrt(1-abar) x0
    ptype = getattr(cfg, "prediction_type", "sample")
    if ptype == "sample":
        objective = diff_target
    elif ptype == "epsilon":
        objective = noise
    elif ptype == "v_prediction":
        objective = scheduler.get_velocity(diff_target, noise, t)
    else:
        raise ValueError(f"unsupported prediction_type: {ptype}")
    flow_loss = huber(pred, objective)
    if use_mu and mu is not None:
        aux = F.mse_loss(mu, x0)                      # train mu toward E[Y|history]
        loss = flow_loss + mu_weight * aux
    else:
        aux = torch.zeros((), device=device)
        loss = flow_loss
    return loss, {"loss": loss.detach(), "flow": flow_loss.detach(), "aux": aux.detach()}


@torch.no_grad()
def ddim_sample_next_shared(generator, scheduler, z, cfg: DiUConfig,
                            mu=None, use_mu: bool = False, data_channels: int = 2):
    B, _, Nt, Nc = z.shape
    device = z.device
    scheduler.set_timesteps(cfg.sampling_steps, device=device)
    if cfg.deterministic_init:
        x = torch.zeros(B, data_channels, Nt, Nc, device=device)
    else:
        x = torch.randn(B, data_channels, Nt, Nc, device=device)
    for t in scheduler.timesteps:
        tn = _tn(t, cfg.num_train_timesteps).expand(B).to(device)
        pred = generator(x, z, tn, tn)
        x = scheduler.step(pred, t, x, eta=cfg.ddim_eta).prev_sample
    if use_mu and mu is not None:
        x = x + mu                                    # reconstruct x0 = mu + residual
    return x


@torch.no_grad()
def ddim_ar_predict_shared(encoder, generator, scheduler, past, num_future,
                           cfg: DiUConfig, use_mu: bool = False, data_channels: int = 2) -> torch.Tensor:
    """Autoregressive DDIM rollout on the shared backbone -> [B, num_future, 2, Nt, Nc]."""
    was = (encoder.training, generator.training)
    encoder.eval(); generator.eval()
    history = past
    preds = []
    for _ in range(num_future):
        z, mu = encoder(history, return_mu=True)
        nxt = ddim_sample_next_shared(generator, scheduler, z, cfg,
                                      mu=(mu if use_mu else None), use_mu=use_mu,
                                      data_channels=data_channels)
        preds.append(nxt)
        history = torch.cat([history, nxt.unsqueeze(1)], dim=1)
    encoder.train(was[0]); generator.train(was[1])
    return torch.stack(preds, dim=1)
