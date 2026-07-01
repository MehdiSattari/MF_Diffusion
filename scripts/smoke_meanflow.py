"""End-to-end smoke test of the MeanFlow objective on real Sionna CSI.

Runs a few optimizer steps on generated data and prints the loss so you can see
it move. Run on Alvis:  python -m scripts.smoke_meanflow
"""

import torch
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset
from mf_csi.models import TemporalEncoder, UNetGenerator
from mf_csi.meanflow import meanflow_loss


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = CSIStreamDataset(cfg.data, batch_size=16, steps_per_epoch=20)
    loader = DataLoader(ds, batch_size=None)

    enc = TemporalEncoder(cfg.encoder).to(device)
    gen = UNetGenerator(cfg.generator).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(gen.parameters()), lr=1e-3)

    enc.train(); gen.train()
    for i, batch in enumerate(loader):
        past = batch["past"].to(device)
        future = batch["future"].to(device)
        loss, metrics = meanflow_loss(enc, gen, past, future, cfg.meanflow)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 5 == 0 or i == 19:
            print(f"step {i:2d} | loss {metrics['loss'].item():.4f} "
                  f"| mse {metrics['mse'].item():.4f} "
                  f"| frac(r!=t) {metrics['frac_r_neq_t'].item():.2f}")

    print("OK: meanflow training smoke test passed (loss should trend down).")


if __name__ == "__main__":
    main()
