"""PyTorch dataset wrappers around the Sionna CDL generator.

We expose CSI as ``(H_past, H_future)`` pairs:

    H_past   : [B, Np, 2, Nt, Nc]   (the conditioning history, "X")
    H_future : [B, Nf, 2, Nt, Nc]   (the prediction target,   "Y")

Because Sionna generates a whole batch in one shot, the dataset yields *full
batches* rather than single samples. Use ``DataLoader(dataset, batch_size=None)``
so PyTorch passes the batch through unchanged.

Two flavours:
  * :class:`CSIStreamDataset`  -- infinite on-the-fly stream for training.
  * :func:`make_fixed_eval_set` -- generate a fixed set once, for reproducible
    validation/test NMSE.

Normalization note: the paper scales the whole dataset to [0, 1] with global
min-max. On an infinite stream we have no global stats, so we normalize
per-sample. The scaling factors are returned alongside the data so predictions
can be mapped back to the physical scale for NMSE. We will revisit whether a
fixed global scale (estimated once) is preferable when we wire up the diffusion
model in Step 4.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..config import DataConfig
from .sionna_cdl import CDLChannelGenerator


def _normalize(csi: np.ndarray, mode: str, global_ab=None):
    """Normalize a batch [B, T, 2, Nt, Nc]. Returns (csi_norm, stats) where stats
    holds the parameters needed to invert the transform.

    global_ab = (a, b): fixed global min/max scalars, required for the "global_*"
    modes (matches the paper's single dataset-wide min-max)."""
    B = csi.shape[0]
    flat = csi.reshape(B, -1)
    if mode == "global_minmax11":
        a, b = global_ab
        scale = (b - a) if (b - a) > 1e-12 else 1.0
        out = 2.0 * (csi - a) / scale - 1.0
        stats = {"mode": "global_minmax11", "a": float(a), "b": float(b)}
        return out.astype(np.float32), stats
    if mode == "global_std":
        mu, sd = global_ab
        sd = sd if sd > 1e-12 else 1.0
        out = (csi - mu) / sd
        stats = {"mode": "global_std", "mu": float(mu), "sd": float(sd)}
        return out.astype(np.float32), stats
    if mode == "minmax":
        mn = flat.min(axis=1)
        mx = flat.max(axis=1)
        scale = np.where((mx - mn) > 1e-12, mx - mn, 1.0)
        out = (csi - mn[:, None, None, None, None]) / scale[:, None, None, None, None]
        stats = {"mode": "minmax", "min": mn, "scale": scale}
    elif mode == "minmax11":
        mn = flat.min(axis=1)
        mx = flat.max(axis=1)
        scale = np.where((mx - mn) > 1e-12, mx - mn, 1.0)
        out = 2.0 * (csi - mn[:, None, None, None, None]) / scale[:, None, None, None, None] - 1.0
        stats = {"mode": "minmax11", "min": mn, "scale": scale}
    elif mode == "std":
        mu = flat.mean(axis=1)
        sd = flat.std(axis=1)
        sd = np.where(sd > 1e-12, sd, 1.0)
        out = (csi - mu[:, None, None, None, None]) / sd[:, None, None, None, None]
        stats = {"mode": "std", "mean": mu, "scale": sd}
    elif mode == "none":
        out = csi
        stats = {"mode": "none"}
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return out.astype(np.float32), stats


def denormalize(x: torch.Tensor, stats: Dict) -> torch.Tensor:
    """Invert :func:`_normalize` for a tensor [B, T, 2, Nt, Nc]."""
    mode = stats["mode"]
    if mode == "global_minmax11":
        a, b = stats["a"], stats["b"]
        return (x + 1.0) * 0.5 * (b - a) + a
    if mode == "global_std":
        return x * stats["sd"] + stats["mu"]
    if mode == "minmax":
        mn = torch.as_tensor(stats["min"], device=x.device, dtype=x.dtype)
        sc = torch.as_tensor(stats["scale"], device=x.device, dtype=x.dtype)
        return x * sc[:, None, None, None, None] + mn[:, None, None, None, None]
    if mode == "minmax11":
        mn = torch.as_tensor(stats["min"], device=x.device, dtype=x.dtype)
        sc = torch.as_tensor(stats["scale"], device=x.device, dtype=x.dtype)
        return (x + 1.0) * 0.5 * sc[:, None, None, None, None] + mn[:, None, None, None, None]
    if mode == "std":
        mu = torch.as_tensor(stats["mean"], device=x.device, dtype=x.dtype)
        sc = torch.as_tensor(stats["scale"], device=x.device, dtype=x.dtype)
        return x * sc[:, None, None, None, None] + mu[:, None, None, None, None]
    return x


def _split_batch(csi: np.ndarray, cfg: DataConfig, stats: Dict) -> Dict:
    """Split [B, T, ...] into past/future and convert to torch tensors."""
    past = torch.from_numpy(csi[:, : cfg.num_past])
    future = torch.from_numpy(csi[:, cfg.num_past :])
    return {"past": past, "future": future, "stats": stats}


class CSIStreamDataset(IterableDataset):
    """Infinite stream of on-the-fly CSI batches for training.

    Yields dicts with keys: ``past`` [B, Np, 2, Nt, Nc], ``future``
    [B, Nf, 2, Nt, Nc], and ``stats`` (normalization parameters).
    """

    def __init__(self, cfg: DataConfig, batch_size: int, steps_per_epoch: Optional[int] = None,
                 global_ab=None):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch  # None -> truly infinite
        self.global_ab = global_ab              # (a, b) for global_* normalization
        self._gen: Optional[CDLChannelGenerator] = None

    def _ensure_gen(self):
        # Lazily build the generator inside the worker process so TF is
        # initialised per-worker (and after fork).
        if self._gen is None:
            self._gen = CDLChannelGenerator(self.cfg)

    def __iter__(self) -> Iterator[Dict]:
        self._ensure_gen()
        count = 0
        while self.steps_per_epoch is None or count < self.steps_per_epoch:
            csi = self._gen.generate(self.batch_size)
            csi, stats = _normalize(csi, self.cfg.normalization, self.global_ab)
            yield _split_batch(csi, self.cfg, stats)
            count += 1


def make_fixed_eval_set(cfg: DataConfig, num_samples: int, batch_size: int,
                        global_ab=None) -> list:
    """Generate a fixed list of evaluation batches once (reproducible)."""
    gen = CDLChannelGenerator(cfg)
    batches = []
    remaining = num_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        csi = gen.generate(b)
        csi, stats = _normalize(csi, cfg.normalization, global_ab)
        batches.append(_split_batch(csi, cfg, stats))
        remaining -= b
    return batches


def estimate_global_minmax(cfg: DataConfig, num_samples: int = 2000, batch_size: int = 256):
    """Estimate a single global (min, max) over raw CSI, for global_* normalization.
    Matches the paper's dataset-wide min-max (fit once, applied to train + eval)."""
    gen = CDLChannelGenerator(cfg)
    a, b = np.inf, -np.inf
    remaining = num_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        csi = gen.generate(n)
        a = min(a, float(csi.min()))
        b = max(b, float(csi.max()))
        remaining -= n
    return a, b


def estimate_global_std(cfg: DataConfig, num_samples: int = 4000, batch_size: int = 256):
    """Estimate a single global (mean, std) over raw CSI, for global_std normalization.
    One fixed scale (fit once, applied to train + eval) -> unit-variance data with no
    per-sample leakage. Pairs with source_std=1 for the MeanFlow informative prior."""
    gen = CDLChannelGenerator(cfg)
    n_tot, s1, s2 = 0, 0.0, 0.0
    remaining = num_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        x = gen.generate(n).astype(np.float64)
        s1 += float(x.sum()); s2 += float((x * x).sum()); n_tot += x.size
        remaining -= n
    mu = s1 / n_tot
    var = max(s2 / n_tot - mu * mu, 0.0)
    return float(mu), float(np.sqrt(var))
