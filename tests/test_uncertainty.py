"""Sanity tests for the UQ + downstream metrics (synthetic, no torch grad needed).

A properly-calibrated ensemble (truth is an independent draw from the same predictive
distribution) should give coverage ~ nominal and spread ~ skill; SE with perfect CSI
should upper-bound SE from a noisy prediction.

Run:  python -m tests.test_uncertainty
"""

import torch

from mf_csi.uncertainty import (crps, coverage, spread_skill, ensemble_mean_nmse,
                                spectral_efficiency, outage_rate,
                                outage_operating_curve, goodput_at_outage)


def test_calibrated_ensemble():
    torch.manual_seed(0)
    B, Nf, Nt, Nc, K = 64, 10, 16, 16, 100
    s = 0.5
    m = torch.randn(B, Nf, 2, Nt, Nc)                  # predictive mean
    samples = m.unsqueeze(0) + s * torch.randn(K, B, Nf, 2, Nt, Nc)
    y = m + s * torch.randn(B, Nf, 2, Nt, Nc)          # truth ~ same predictive dist

    _, cov = coverage(samples, y, level=0.9)
    assert 0.86 <= cov.item() <= 0.94, f"coverage {cov.item():.3f} not ~0.9 (miscalibrated metric)"

    spread, skill = spread_skill(samples, y)
    ratio = (spread.mean() / skill.mean()).item()
    assert 0.9 <= ratio <= 1.1, f"spread/skill {ratio:.3f} should be ~1 for a calibrated ensemble"

    ps_crps, ov_crps = crps(samples, y)
    assert ps_crps.shape == (Nf,) and torch.isfinite(ov_crps) and ov_crps > 0

    ps_n, ov_n = ensemble_mean_nmse(samples, y)
    assert ps_n.shape == (Nf,) and torch.isfinite(ov_n)
    print(f"OK UQ | coverage {cov.item():.3f} | spread/skill {ratio:.3f} | "
          f"CRPS {ov_crps.item():.4f} | mean-NMSE {ov_n.item():.4f}")


def test_spectral_efficiency_bound():
    torch.manual_seed(0)
    B, Nf, Nt, Nc = 32, 10, 16, 16
    true = torch.randn(B, Nf, 2, Nt, Nc)
    pred = true + 0.3 * torch.randn(B, Nf, 2, Nt, Nc)  # noisy prediction
    se_pred, se_perfect = spectral_efficiency(pred, true, snr_db=20.0)
    assert se_pred.shape == (Nf,) and se_perfect.shape == (Nf,)
    assert se_perfect.mean().item() >= se_pred.mean().item(), "perfect-CSI SE must upper-bound predicted"
    print(f"OK SE | pred {se_pred.mean().item():.3f} <= perfect {se_perfect.mean().item():.3f} b/s/Hz")


def test_operating_curve_not_gameable():
    """A model that UNDER-predicts the channel must not win goodput at equal outage:
    the operating curve penalises conservative bias (the flaw in the single-point metric)."""
    torch.manual_seed(0)
    B, Nf, Nt, Nc, K = 48, 10, 16, 16, 60
    true = torch.randn(B, Nf, 2, Nt, Nc)
    good = true.unsqueeze(0) + 0.3 * torch.randn(K, B, Nf, 2, Nt, Nc)   # accurate ensemble
    biased = 0.5 * good                                                 # systematically under-predicts
    og, gg = outage_operating_curve(good, true, 20.0)
    ob, gb = outage_operating_curve(biased, true, 20.0)
    g_good = goodput_at_outage(og, gg, 0.1)
    g_biased = goodput_at_outage(ob, gb, 0.1)
    assert g_good > g_biased, f"under-predictor should not win goodput@outage ({g_biased:.2f} vs {g_good:.2f})"
    print(f"OK operating curve | goodput@0.1: accurate {g_good:.2f} > under-predictor {g_biased:.2f}")


def test_outage_calibration():
    torch.manual_seed(0)
    B, Nf, Nt, Nc, K = 64, 10, 16, 16, 200
    # iid channels: predicted samples and truth from the same distribution -> a
    # calibrated predictor should achieve empirical outage ~ epsilon.
    m = torch.randn(B, Nf, 2, Nt, Nc)
    samples = m.unsqueeze(0) + 0.5 * torch.randn(K, B, Nf, 2, Nt, Nc)
    truth = m + 0.5 * torch.randn(B, Nf, 2, Nt, Nc)
    eps = 0.1
    gp, out = outage_rate(samples, truth, snr_db=20.0, epsilon=eps)
    assert gp.shape == (Nf,) and out.shape == (Nf,)
    assert abs(out.mean().item() - eps) < 0.05, f"empirical outage {out.mean().item():.3f} vs target {eps}"
    print(f"OK outage | empirical {out.mean().item():.3f} ~ target {eps} | goodput {gp.mean().item():.3f} b/s/Hz")


if __name__ == "__main__":
    test_calibrated_ensemble()
    test_spectral_efficiency_bound()
    test_outage_calibration()
    test_operating_curve_not_gameable()
    print("\nUncertainty/downstream metric tests passed.")
