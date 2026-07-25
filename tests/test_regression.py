"""Shape + gradient test for the JointRegressor ConvLSTM baseline.

Run (needs torch, no Sionna):  python -m tests.test_regression
"""

import torch

from mf_csi.config import Config
from mf_csi.models import JointRegressor


def test_joint_regressor_shapes_and_grad():
    cfg = Config()
    model = JointRegressor(cfg.regression)
    B = 3
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    past = torch.randn(B, Np, 2, Nt, Nc, requires_grad=True)
    pred = model(past)
    assert pred.shape == (B, Nf, 2, Nt, Nc), pred.shape        # all frames at once
    pred.pow(2).mean().backward()
    assert past.grad is not None and torch.isfinite(past.grad).all()
    n = sum(p.numel() for p in model.parameters())
    print(f"OK JointRegressor -> {tuple(pred.shape)} | params {n/1e6:.3f}M")


if __name__ == "__main__":
    test_joint_regressor_shapes_and_grad()
    print("\nJointRegressor test passed.")
