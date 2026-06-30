"""Logic tests that do NOT require Sionna/TensorFlow.

We monkeypatch the channel generator with a mock that returns random arrays of
the correct shape, so we can verify the windowing + normalization + tensor
conversion independently of the (heavy) Sionna stack.

Run:  python -m tests.test_dataset_mock
"""

import numpy as np
import torch

from mf_csi.config import DataConfig
from mf_csi.data import dataset as ds_mod
from mf_csi.data.dataset import _normalize, denormalize, _split_batch


def test_normalize_roundtrip():
    cfg = DataConfig()
    rng = np.random.default_rng(0)
    csi = rng.standard_normal((4, cfg.seq_len, 2, cfg.num_bs_ant,
                               cfg.num_subcarriers_used)).astype(np.float32)
    for mode in ("minmax", "std", "none"):
        norm, stats = _normalize(csi, mode)
        back = denormalize(torch.from_numpy(norm), stats).numpy()
        assert np.allclose(back, csi, atol=1e-4), f"roundtrip failed for {mode}"
        if mode == "minmax":
            assert norm.min() >= -1e-5 and norm.max() <= 1 + 1e-5
    print("OK: normalize roundtrip (minmax/std/none)")


def test_split_shapes():
    cfg = DataConfig()
    csi = np.zeros((3, cfg.seq_len, 2, cfg.num_bs_ant, cfg.num_subcarriers_used),
                   dtype=np.float32)
    out = _split_batch(csi, cfg, {"mode": "none"})
    assert tuple(out["past"].shape) == (3, cfg.num_past, 2, cfg.num_bs_ant,
                                        cfg.num_subcarriers_used)
    assert tuple(out["future"].shape) == (3, cfg.num_future, 2, cfg.num_bs_ant,
                                          cfg.num_subcarriers_used)
    print("OK: past/future split shapes")


class _MockGen:
    def __init__(self, cfg):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def generate(self, batch_size):
        return self._rng.standard_normal(
            (batch_size,) + self.cfg.sample_shape
        ).astype(np.float32)


def test_stream_with_mock(monkeypatch=None):
    cfg = DataConfig()
    # Swap the real generator for the mock.
    ds_mod.CDLChannelGenerator = _MockGen
    stream = ds_mod.CSIStreamDataset(cfg, batch_size=5, steps_per_epoch=3)
    n = 0
    for batch in stream:
        assert tuple(batch["past"].shape) == (5, cfg.num_past, 2,
                                              cfg.num_bs_ant, cfg.num_subcarriers_used)
        n += 1
    assert n == 3, "steps_per_epoch not respected"
    print("OK: CSIStreamDataset yields correct batches (mock generator)")


if __name__ == "__main__":
    test_normalize_roundtrip()
    test_split_shapes()
    test_stream_with_mock()
    print("\nAll mock-based data tests passed.")
