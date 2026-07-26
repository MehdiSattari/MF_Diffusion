"""Sanity tests for the UQ + downstream metrics (synthetic, no torch grad needed).

A properly-calibrated ensemble (truth is an independent draw from the same predictive
distribution) should give coverage ~ nominal and spread ~ skill; SE with perfect CSI
should upper-bound SE from a noisy prediction.

Run:  python -m tests.test_uncertainty
"""

import torch

from mf_csi.uncertainty import (crps, coverage, spread_skill, ensemble_mean_nmse,
                                spectral_efficiency, outage_rate)


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
    print("\nUncertainty/downstream metric tests passed.")
