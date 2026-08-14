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


def per_sample_nmse(samples: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """NMSE of an INDIVIDUAL sample, averaged over the K draws (the 'normal', point-to-point
    NMSE of a single prediction). Unlike the ensemble mean, this does not average out sample
    noise, so it measures how good a single generated frame is -- and it should improve as
    more sampling steps make each draw more faithful. samples [K,B,Nf,2,Nt,Nc], y [B,...]."""
    err = (samples - y.unsqueeze(0)).pow(2).flatten(3).sum(-1)      # [K,B,Nf]
    power = y.pow(2).flatten(2).sum(-1).clamp_min(1e-12)            # [B,Nf]
    nmse = err / power.unsqueeze(0)                                 # [K,B,Nf]
    per_step = nmse.mean(dim=(0, 1))                                # [Nf]
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


def calibration_error(samples: torch.Tensor, y: torch.Tensor,
                      levels=(0.1, 0.3, 0.5, 0.7, 0.9)) -> float:
    """ECE-style interval calibration error: mean over nominal `levels` of
    |empirical_coverage(level) - level|. Lower is better; 0 == perfectly calibrated.
    Complements the reliability diagram with a single scalar."""
    errs = [abs(float(coverage(samples, y, lv)[1]) - lv) for lv in levels]
    return sum(errs) / len(errs)


def rank_counts(samples: torch.Tensor, y: torch.Tensor, n_bins: int = 10):
    """PIT / verification-rank histogram bin counts (unnormalized) for one batch.

    For each scalar coefficient, the rank r = #{ensemble members < truth} / K in [0,1]
    is binned into `n_bins`. A calibrated ensemble gives a FLAT histogram; an
    overconfident (under-dispersed) one gives a U-shape (mass piling at the ends).
    Returns a length-`n_bins` tensor of counts; sum across batches, then normalize."""
    K = samples.shape[0]
    r = (samples < y.unsqueeze(0)).sum(dim=0).float() / K       # [B,Nf,2,Nt,Nc], in [0,1]
    idx = (r.clamp(0.0, 1.0 - 1e-6) * n_bins).long().clamp(0, n_bins - 1)
    return torch.bincount(idx.flatten(), minlength=n_bins).float()


def rank_uniformity(hist_norm) -> float:
    """Scalar summary of a normalized rank histogram: total-variation distance from
    uniform, sum_b |h_b - 1/n_bins|. 0 == perfectly flat (calibrated); larger == more
    U-shaped (overconfident)."""
    import torch as _t
    h = _t.as_tensor(hist_norm, dtype=_t.float32)
    n = h.numel()
    return float((h - 1.0 / n).abs().sum())


def spread_skill(samples: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-step ensemble spread (RMS std over K) and skill (RMSE of the ensemble mean).
    A calibrated ensemble has spread/skill ~ sqrt(K/(K+1)) (=> ~0.98 for K=30, i.e. ~1);
    a ratio well below 1 means over-confident. Returns (spread [Nf], skill [Nf])."""
    mean = samples.mean(dim=0)
    var = samples.var(dim=0, unbiased=True)                      # [B,Nf,2,Nt,Nc]
    spread = _reduce_steps(var).sqrt()
    sk = _reduce_steps((mean - y).pow(2)).sqrt()
    return spread, sk


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """[.., Nf, 2, Nt, Nc] -> complex [.., Nf, Nt, Nc]."""
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :])


def outage_rate(samples: torch.Tensor, true: torch.Tensor, snr_db: float = 20.0,
                epsilon: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Risk-aware link adaptation: pick a rate to meet a target OUTAGE probability.

    Using the K predicted channels the transmitter forms the predicted distribution of
    the achievable MR rate C = log2(1 + snr*||h||^2) and selects R = epsilon-quantile of
    that distribution (the rate it is 1-epsilon confident it can support). Outage occurs
    if the TRUE channel cannot support R. A CALIBRATED generative model achieves empirical
    outage ~ epsilon with high goodput; a deterministic model (K=1) cannot control outage.

    samples [K,B,Nf,2,Nt,Nc], true [B,Nf,2,Nt,Nc] -> (goodput [Nf], empirical_outage [Nf]).
    """
    snr = 10.0 ** (snr_db / 10.0)
    hp = _to_complex(samples)                                   # [K,B,Nf,Nt,Nc]
    ht = _to_complex(true)                                      # [B,Nf,Nt,Nc]
    c_pred = torch.log2(1.0 + snr * (hp.abs() ** 2).sum(dim=3))  # [K,B,Nf,Nc]
    c_true = torch.log2(1.0 + snr * (ht.abs() ** 2).sum(dim=2))  # [B,Nf,Nc]
    if c_pred.shape[0] == 1:
        R = c_pred[0]                                          # point predictor: no distribution
    else:
        R = torch.quantile(c_pred, epsilon, dim=0)            # epsilon-outage rate
    success = (c_true >= R).float()
    goodput = (R * success).mean(dim=(0, 2))                   # [Nf]
    outage = (1.0 - success).mean(dim=(0, 2))                  # [Nf]
    return goodput, outage


def rate_quantities(samples: torch.Tensor, true: torch.Tensor, snr_db: float = 20.0):
    """Achievable MR rates. Returns (c_pred [K,B,Nf,Nc], c_true [B,Nf,Nc])."""
    snr = 10.0 ** (snr_db / 10.0)
    hp = _to_complex(samples); ht = _to_complex(true)
    c_pred = torch.log2(1.0 + snr * (hp.abs() ** 2).sum(dim=3))
    c_true = torch.log2(1.0 + snr * (ht.abs() ** 2).sum(dim=2))
    return c_pred, c_true


def outage_global(samples: torch.Tensor, true: torch.Tensor, snr_db: float = 20.0,
                  epsilon: float = 0.1) -> Tuple[float, float, float]:
    """Case 1 (single GLOBAL rate). R = epsilon-quantile of the model's pooled predicted
    rate distribution; then achieved outage = P(c_true < R), goodput = R*(1-achieved).
    A biased/high-variance model cannot allocate an aggregate outage budget unevenly here."""
    c_pred, c_true = rate_quantities(samples, true, snr_db)
    R = torch.quantile(c_pred.reshape(-1), epsilon)
    achieved = (c_true < R).float().mean()
    goodput = R * (1.0 - achieved)
    return float(R), float(achieved), float(goodput)


def selected_rates(samples: torch.Tensor, true: torch.Tensor, snr_db: float = 20.0,
                   epsilon: float = 0.1):
    """Per-coefficient selected rate R_i (epsilon-quantile over samples) + c_true, flattened,
    for diagnosing whether a model is aggressive (R shifted high) or high-variance (R spread)."""
    c_pred, c_true = rate_quantities(samples, true, snr_db)
    R = torch.quantile(c_pred, epsilon, dim=0)              # [B,Nf,Nc]
    return R.reshape(-1), c_true.reshape(-1)


def crps_rate(samples: torch.Tensor, true: torch.Tensor, snr_db: float = 20.0
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """CRPS of the achievable MR rate -- a PROPER, un-gameable downstream score that ties
    uncertainty quality to communications performance (as suggested in review).

    For each coefficient the model induces a distribution of achievable rates
    C_k = log2(1 + snr * ||h_pred_k||^2); we score it against the true achievable rate
    C_true = log2(1 + snr * ||h_true||^2) with CRPS = E|C-C_true| - 1/2 E|C-C'|. Lower is
    better. Unlike goodput, no aggregate-outage budget can be spent unevenly, and unlike
    NMSE it rewards predicting the *distribution* of rate, not just the mean.

    samples [K,B,Nf,2,Nt,Nc], true [B,Nf,2,Nt,Nc] -> (per_step [Nf], overall).
    """
    c_pred, c_true = rate_quantities(samples, true, snr_db)      # [K,B,Nf,Nc], [B,Nf,Nc]
    K = c_pred.shape[0]
    term1 = (c_pred - c_true.unsqueeze(0)).abs().mean(dim=0)     # E|C - C_true|  [B,Nf,Nc]
    xs, _ = torch.sort(c_pred, dim=0)
    k = torch.arange(1, K + 1, device=c_pred.device, dtype=c_pred.dtype)
    w = (2.0 * k - K - 1.0).view(K, 1, 1, 1)
    exx = (2.0 / (K * K)) * (w * xs).sum(dim=0)                  # E|C - C'|      [B,Nf,Nc]
    crps_c = term1 - 0.5 * exx
    per_step = crps_c.mean(dim=(0, 2))                           # [Nf]
    return per_step, per_step.mean()


def outage_operating_curve(samples: torch.Tensor, true: torch.Tensor, snr_db: float = 20.0,
                           q_grid: "torch.Tensor | None" = None
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Goodput-vs-outage operating curve. NOTE: this is the *per-coefficient* rate with
    *aggregate* outage (a distinct R per (b,f,subcarrier), outage averaged over all).
    This definition is NOT bias-robust: a high-variance / optimistically-biased model can
    raise average goodput by spending an aggregate outage budget unevenly across
    coefficients. Use `outage_global` (single global rate) for a bias-robust scalar, and
    the calibration scores (coverage, CRPS, spread-skill) for the reliable verdict.

    For each target quantile q, R_q = q-quantile of the predicted achievable-rate
    distribution; achieved outage and goodput are measured on the true channel.
    samples [K,B,Nf,2,Nt,Nc], true [B,Nf,2,Nt,Nc] -> (achieved_outage [Q], goodput [Q]).
    """
    if q_grid is None:
        q_grid = torch.linspace(0.02, 0.6, 30)
    snr = 10.0 ** (snr_db / 10.0)
    hp = _to_complex(samples)                                   # [K,B,Nf,Nt,Nc]
    ht = _to_complex(true)                                      # [B,Nf,Nt,Nc]
    c_pred = torch.log2(1.0 + snr * (hp.abs() ** 2).sum(dim=3)) # [K,B,Nf,Nc]
    c_true = torch.log2(1.0 + snr * (ht.abs() ** 2).sum(dim=2)) # [B,Nf,Nc]
    q_grid = q_grid.to(c_pred.device, c_pred.dtype)
    R = torch.quantile(c_pred, q_grid, dim=0)                   # [Q,B,Nf,Nc]
    success = (c_true.unsqueeze(0) >= R).float()
    achieved_outage = (1.0 - success).mean(dim=(1, 2, 3))       # [Q]
    goodput = (R * success).mean(dim=(1, 2, 3))                 # [Q]
    return achieved_outage, goodput


def goodput_at_outage(achieved_outage: torch.Tensor, goodput: torch.Tensor,
                      epsilon: float = 0.1) -> float:
    """Interpolate the operating curve to the goodput where achieved outage = epsilon.
    The single fair scalar: throughput each model can guarantee at the target reliability."""
    o = achieved_outage.detach().cpu().numpy()
    g = goodput.detach().cpu().numpy()
    order = o.argsort()
    return float(__import__("numpy").interp(epsilon, o[order], g[order]))


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
