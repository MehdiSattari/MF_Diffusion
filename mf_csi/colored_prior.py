"""Channel-shaped (colored) source prior for the informative-prior generators.

Motivation: the residual  r = H - mu  of a wireless channel is sparse in the
angle-delay domain, so an isotropic source  sigma^2 I  wastes most of its variance
budget in directions the channel never occupies. We instead color the source with a
structured residual covariance R:

    S = sg(mu) + sigma * color(eps),   eps ~ N(0, I),   Cov[color(eps)] ~ R / mean-power.

R is approximated as circulant (wide-sense stationary across antennas and subcarriers),
hence DIAGONAL in the 2D-DFT (angle-delay) domain with eigenvalues = the average power
spectral density (PSD) of the residuals. Coloring is then a variance-preserving
elementwise scaling in the DFT domain -- essentially free at inference:

    color(eps) = IDFT2( sqrt(psd_norm) * DFT2(eps) ),   mean(psd_norm) = 1.

Shape (psd_norm) is decoupled from magnitude (sigma), so sigma can still be tuned or made
SNR/horizon dependent. Tensors are the usual real layout [.., 2, Nt, Nc] (2 = re/im).
"""
from __future__ import annotations
import torch


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """[.., 2, Nt, Nc] real/imag -> complex [.., Nt, Nc]."""
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :])


def to_real2(c: torch.Tensor) -> torch.Tensor:
    """complex [.., Nt, Nc] -> [.., 2, Nt, Nc] real/imag."""
    return torch.stack([c.real, c.imag], dim=-3)


def residual_psd(residual2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Estimate the normalized angle-delay PSD from residuals.

    residual2: [B, 2, Nt, Nc] (real/imag of H - mu). Returns psd_norm [Nt, Nc] with
    mean 1 (so coloring preserves total variance)."""
    r = to_complex(residual2)                                  # [B, Nt, Nc] complex
    Rf = torch.fft.fft2(r, norm="ortho")                       # angle-delay domain
    psd = (Rf.abs() ** 2).mean(dim=0)                          # [Nt, Nc], >= 0
    psd = psd / psd.mean().clamp_min(eps)                      # normalize to unit mean power
    return psd


def color(eps2: torch.Tensor, psd_norm: torch.Tensor) -> torch.Tensor:
    """Color white real/imag noise with sqrt(psd) in the 2D-DFT domain.

    eps2: [.., 2, Nt, Nc] white ~ N(0, I). psd_norm: [Nt, Nc] (mean 1). Returns the same
    shape, wide-sense-stationary colored noise with unit total variance (Parseval)."""
    e = to_complex(eps2)                                       # [.., Nt, Nc] complex
    Ef = torch.fft.fft2(e, norm="ortho")
    Ef = Ef * torch.sqrt(psd_norm.to(Ef.real.dtype))
    out = torch.fft.ifft2(Ef, norm="ortho")
    return to_real2(out).to(eps2.dtype)
