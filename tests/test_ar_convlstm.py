"""Shape + gradient + AR-rollout test for the standalone AR ConvLSTM baseline."""
import torch
from mf_csi.config import Config
from mf_csi.models import ARConvLSTM
from mf_csi.inference import ar_convlstm_predict


def test_ar_convlstm():
    cfg = Config()
    model = ARConvLSTM(cfg.ar_convlstm)
    B = 2
    Np, Nf = cfg.data.num_past, cfg.data.num_future
    Nt, Nc = cfg.data.num_bs_ant, cfg.data.num_subcarriers_used
    past = torch.randn(B, Np, 2, Nt, Nc, requires_grad=True)
    nxt = model(past)
    assert nxt.shape == (B, 2, Nt, Nc), nxt.shape
    nxt.pow(2).mean().backward()
    assert past.grad is not None and torch.isfinite(past.grad).all()
    pred = ar_convlstm_predict(model, past.detach(), Nf)
    assert pred.shape == (B, Nf, 2, Nt, Nc), pred.shape
    n = sum(p.numel() for p in model.parameters())
    print(f"OK AR-ConvLSTM | next {tuple(nxt.shape)} | rollout {tuple(pred.shape)} | params {n/1e6:.3f}M")


if __name__ == "__main__":
    test_ar_convlstm()
    print("\nAR ConvLSTM test passed.")
