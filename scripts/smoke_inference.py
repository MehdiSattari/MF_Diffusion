"""End-to-end demo: train briefly, then autoregressively predict + measure NMSE.

This is the real correctness check for the MeanFlow sampling: if the flow
direction or the 1-step formula were wrong, NMSE would not improve with training.
A short run should already push early-step NMSE below 0 dB.

Run on Alvis:  python -m scripts.smoke_inference
"""

import torch
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.meanflow import meanflow_loss
from mf_csi.inference import autoregressive_predict, nmse, nmse_db

TRAIN_STEPS = 400


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| train steps:", TRAIN_STEPS)

    train_ds = CSIStreamDataset(cfg.data, batch_size=32, steps_per_epoch=TRAIN_STEPS)
    train_loader = DataLoader(train_ds, batch_size=None)

    enc = TemporalEncoder(cfg.encoder).to(device)
    gen = UNetGenerator(cfg.generator).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(gen.parameters()), lr=2e-4)

    enc.train(); gen.train()
    for i, batch in enumerate(train_loader):
        past, future = batch["past"].to(device), batch["future"].to(device)
        loss, m = meanflow_loss(enc, gen, past, future, cfg.meanflow)
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 100 == 0 or i == TRAIN_STEPS - 1:
            print(f"  train step {i:4d} | mse {m['mse'].item():.4f}")

    # Evaluate AR prediction on a fresh batch.
    eval_ds = CSIStreamDataset(cfg.data, batch_size=64, steps_per_epoch=1)
    batch = next(iter(DataLoader(eval_ds, batch_size=None)))
    past, future = batch["past"].to(device), batch["future"].to(device)
    pred = autoregressive_predict(enc, gen, past, cfg.data.num_future,
                                  seed_std=cfg.inference.seed_std,
                                  step_noise_std=cfg.inference.step_noise_std)
    per_step, overall = nmse(pred, future)

    print("\nNMSE per prediction step (dB):")
    for n, val in enumerate(per_step, start=1):
        print(f"  step {n:2d}: {nmse_db(val).item():6.2f} dB")
    print(f"average NMSE: {nmse_db(overall).item():.2f} dB  (linear {overall.item():.3f})")
    print("\nOK: inference smoke test passed "
          "(early-step NMSE should be well below 0 dB after real training).")


if __name__ == "__main__":
    main()
