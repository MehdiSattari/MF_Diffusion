"""MeanFlow training objective for CSI prediction (the paper's Algorithm 2).

Given a batch of history/future CSI pairs, this computes the MeanFlow loss for
the DiU generator conditioned on the ConvLSTM latent Z:

  1. Optionally noise-augment the history X -> X~ (CSI-estimation-error model).
  2. Z = encoder(X~).
  3. Sample (r, t) with t >= r; Y = next future frame.
  4. Flow interpolant  H^t = (1-t) Y + t eps ;  velocity  v = eps - Y.
  5. u from a normal forward pass (keeps the reverse-mode graph to BOTH the
     generator and the encoder-through-Z); d/dt u from a SEPARATE forward-mode-AD
     pass with time tangent (v, 0, 1) on (H^t, r, t) — value only.
  6. Target  u_tgt = v - (t - r) d/dt u  (stop-gradient).
  7. Adaptively-weighted loss  sg(w) * ||u - u_tgt||^2,  w = 1/(||.||^2 + c)^p.

Two design notes:
  * forward_ad (dual tensors), not torch.func.jvp: torch.func.jvp treats the
    closed-over parameters as constants, so no weight gradients would flow.
  * Separate passes for u and d/dt u: computing both in ONE dual pass drops the
    reverse-mode edge from u back to Z (a plain, non-dual input), which silently
    zeroes the ENCODER gradient. Since the target is stop-gradient, d/dt u needs
    no backward graph, so a second detached forward-AD pass is the clean fix.
    (Cost: one extra generator forward per step — fuseable later if needed.)
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.autograd.forward_ad as fwAD

from .config import MeanFlowConfig


def sample_r_t(batch_size: int, cfg: MeanFlowConfig, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample time pairs (r, t) in (0, 1) with t >= r. A fraction
    (1 - ratio_r_not_equal_t) is forced to r == t (reduces to flow matching)."""
    if cfg.time_sampler == "lognorm":
        n = torch.randn(batch_size, 2, device=device) * cfg.lognorm_std + cfg.lognorm_mean
        samples = torch.sigmoid(n)
    elif cfg.time_sampler == "uniform":
        samples = torch.rand(batch_size, 2, device=device)
    else:
        raise ValueError(f"Unknown time_sampler: {cfg.time_sampler}")
    t = samples.max(dim=1).values
    r = samples.min(dim=1).values
    force_equal = torch.rand(batch_size, device=device) >= cfg.ratio_r_not_equal_t
    r = torch.where(force_equal, t, r)
    return r, t


def augment_history(past: torch.Tensor, cfg: MeanFlowConfig) -> torch.Tensor:
    """Additive CSI-estimation-error noise at a random SNR in [snr_db_min, snr_db_max]:

        X~ = X + sigma * N,   sigma chosen per-sample so that  signal_power / noise_power = SNR.

    This PRESERVES the signal scale, unlike the sqrt(rho)*X + N form: that rescaling
    (up to 10x at 20 dB) makes the autoregressive rollout mix 10x-scaled observed
    frames with 1x-scaled predicted frames, which inverts the NMSE-vs-SNR ordering
    and destabilizes training. Additive noise keeps every frame at its natural scale.
    """
    if not cfg.noise_aug:
        return past
    B = past.shape[0]
    snr_db = torch.empty(B, device=past.device).uniform_(cfg.snr_db_min, cfg.snr_db_max)
    rho = 10.0 ** (snr_db / 10.0)                            # linear SNR [B]
    power = past.pow(2).flatten(1).mean(dim=1)               # mean per-element signal power [B]
    sigma = torch.sqrt(power / rho).view(B, *([1] * (past.dim() - 1)))
    return past + torch.randn_like(past) * sigma


def meanflow_loss(encoder, generator, past: torch.Tensor, future: torch.Tensor,
                  cfg: MeanFlowConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the MeanFlow loss for one batch.

    past:   [B, Np, 2, Nt, Nc]   history (conditioning)
    future: [B, Nf, 2, Nt, Nc]   the next frame future[:, 0] is the target Y
    Returns (loss, metrics).
    """
    device = future.device
    B = future.shape[0]

    # Steps 1-2: conditioning latent from (noise-augmented) history.
    x = augment_history(past, cfg)
    z = encoder(x)                                  # [B, Cz, Nt, Nc]

    # Step 3-4: next-frame target, interpolant, instantaneous velocity.
    Y = future[:, 0]                                # [B, 2, Nt, Nc]
    eps = torch.randn_like(Y)
    r, t = sample_r_t(B, cfg, device)
    t_b = t.view(B, 1, 1, 1)
    Ht = (1.0 - t_b) * Y + t_b * eps
    v = eps - Y

    # Step 5a: u with a NORMAL forward pass -> full reverse-mode graph, so the
    # loss backpropagates into BOTH the generator AND the encoder (through Z).
    u = generator(Ht, z, r, t)

    # Step 5b: d/dt u via a SEPARATE forward-mode-AD pass (value only). The target
    # is stop-gradient, so d/dt u needs no backward graph; inputs are detached so
    # this pass builds no gradient path. Tangent (v, 0, 1) on (Ht, r, t) gives
    #   d/dt u = v . d_h u + d_t u.
    with fwAD.dual_level():
        h_dual = fwAD.make_dual(Ht.detach(), v)
        r_dual = fwAD.make_dual(r.detach(), torch.zeros_like(r))
        t_dual = fwAD.make_dual(t.detach(), torch.ones_like(t))
        u_dual = generator(h_dual, z.detach(), r_dual, t_dual)
        dudt = fwAD.unpack_dual(u_dual).tangent
    dudt = (torch.zeros_like(u) if dudt is None else dudt).detach()

    # Step 6: stop-gradient MeanFlow target.
    tr = (t - r).view(B, 1, 1, 1)
    u_tgt = (v - tr * dudt).detach()

    # Step 7: adaptively-weighted squared error.
    err = u - u_tgt
    sq = err.pow(2).flatten(1).mean(dim=1)          # [B] per-sample MSE
    w = 1.0 / (sq.detach() + cfg.loss_eps).pow(cfg.loss_power)
    loss = (w * sq).mean()

    metrics = {
        "loss": loss.detach(),
        "mse": sq.mean().detach(),
        "frac_r_neq_t": (r != t).float().mean().detach(),
    }
    return loss, metrics
