"""Uncertainty-quantification + downstream metrics for generative CSI prediction.

These are the metrics that decide whether the generative residual adds value BEYOND
the conditional mean (mu / regression) -- the point-metric NMSE cannot show this by
construction. Given K samples per input from a generative model:

  * ensemble_mean_nmse : NMSE of the ensemble mean (should ~ regression -- a sanity floor)
  * crps               : Continuous Ranked Probability Score (proper score; calibration+sharpness)
  * coverage           : PICP -- fraction of truth inside the central `level` predictive interval
  * spread_skill       : ensemble spread vs error per step (calibrated -> spread ~ RMSE)
  * spectral_efficiency: downstream MR-precoding rate from predicted vs true CSI

All operate on real tensors laid out as [.., Nf, 2, Nt, Nc] (2 = real/imag). CRPS,
coverage and spread-skill take an ensemble [K, B, Nf, 2, Nt, Nc]; compute them in the
PHYSICAL (denormalized) space for interpretability.
"""

from __future__ import annotations

from typing import Tuple
import torch


def _reduce_steps(x: torch.Tensor) -> torch.Tensor:
    """Mean over everything except the step axis. x: [B, Nf, 2, Nt, Nc] -> [Nf]."""
    return x.mean(dim=(0, 2, 3, 4))


def ensemble_mean_nmse(samples: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """NMSE of the ensemble mean. samples [K,B,Nf,2,Nt,Nc], y [B,Nf,2,Nt,Nc]."""
    mean = samples.mean(dim=0)
    err = (mean - y).pow(2).flatten(2).sum(-1)
    power = y.pow(2).flatten(2).sum(-1).clamp_min(1e-12)
    per_step = (err / power).mean(0)
    return per_step, per_step.mean()


def crps(samples: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample-based CRPS (lower is better).

    CRPS(F, y) = E|X - y| - 1/2 E|X - X'|, estimated per element then averaged.
    The E|X-X'| term uses the O(K log K) sorted identity
        E|X-X'| = (2/K^2) sum_k (2k - K - 1) x_(k).
    samples [K,B,Nf,2,Nt,Nc], y [B,Nf,2,Nt,Nc] -> (per_step [Nf], overall).
    """
    K = samples.shape[0]
    term1 = (samples - y.unsqueeze(0)).abs().mean(dim=0)          # E|X - y|, per element
    xs, _ = torch.sort(samples, dim=0)                           # sort along K
    k = torch.arange(1, K + 1, device=samples.device, dtype=samples.dtype)
    w = (2.0 * k - K - 1.0).view(K, *([1] * (samples.dim() - 1)))
    exx = (2.0 / (K * K)) * (w * xs).sum(dim=0)                  # E|X - X'|, per element
    crps_elem = term1 - 0.5 * exx
    per_step = _reduce_steps(crps_elem)
    return per_step, per_step.mean()


def coverage(samples: torch.Tensor, y: torch.Tensor, level: float = 0.9
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """PICP: fraction of truth within the central `level` predictive interval.
    Well-calibrated -> coverage ~ level. Returns (per_step [Nf], overall)."""
    lo = (1.0 - level) / 2.0
    q = torch.quantile(samples, torch.tensor([lo, 1.0 - lo], device=samples.device,
                                             dtype=samples.dtype), dim=0)   # [2,B,Nf,2,Nt,Nc]
    inside = ((y >= q[0]) & (y <= q[1])).float()
    per_step = _reduce_steps(inside)
    return per_step, per_step.mean()


def spread_skill(samples: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-step ensemble spread (mean std over K) and skill (RMSE of the ensemble mean).
    A calibrated ensemble has spread ~ skill. Returns (spread [Nf], skill [Nf])."""
    mean = samples.mean(dim=0)
    var = samples.var(dim=0, unbiased=True)                      # [B,Nf,2,Nt,Nc]
    spread = _reduce_steps(var).sqrt()
    sk = _reduce_steps((mean - y).pow(2)).sqrt()
    return spread, sk


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """[.., Nf, 2, Nt, Nc] -> complex [.., Nf, Nt, Nc]."""
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :])


def spectral_efficiency(pred: torch.Tensor, true: torch.Tensor, snr_db: float = 20.0
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downstream MR-precoding spectral efficiency (bits/s/Hz), per prediction step.

    Per (sample, step, subcarrier) the channel is a length-Nt vector. A maximum-ratio
    precoder w = h_pred/||h_pred|| is formed from the PREDICTED channel and applied to
    the TRUE channel; SE = log2(1 + snr * |h_true^H w|^2). Also returns the perfect-CSI
    SE (w from the true channel) as the ceiling. pred/true: [B,Nf,2,Nt,Nc] (physical).
    """
    snr = 10.0 ** (snr_db / 10.0)
    hp = _to_complex(pred)                                       # [B,Nf,Nt,Nc]
    ht = _to_complex(true)
    inner = (hp.conj() * ht).sum(dim=2)                          # [B,Nf,Nc]
    num = inner.abs() ** 2
    den = (hp.abs() ** 2).sum(dim=2).clamp_min(1e-12)
    g_pred = num / den
    g_perfect = (ht.abs() ** 2).sum(dim=2)                       # ||h_true||^2 (MR ceiling)
    se_pred = torch.log2(1.0 + snr * g_pred).mean(dim=(0, 2))    # [Nf]
    se_perfect = torch.log2(1.0 + snr * g_perfect).mean(dim=(0, 2))
    return se_pred, se_perfect
