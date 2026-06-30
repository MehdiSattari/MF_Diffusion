"""End-to-end smoke test: Sionna CDL data -> ConvLSTM temporal encoder.

Run on Alvis (needs Sionna + torch):  python -m scripts.smoke_encoder
"""

import torch
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset
from mf_csi.models import TemporalEncoder


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = CSIStreamDataset(cfg.data, batch_size=4, steps_per_epoch=1)
    loader = DataLoader(ds, batch_size=None)
    enc = TemporalEncoder(cfg.encoder).to(device)

    for batch in loader:
        past = batch["past"].to(device)          # [B, Np, 2, Nt, Nc]
        z = enc(past)                            # [B, C_z, Nt, Nc]
        print("past", tuple(past.shape), "-> Z", tuple(z.shape))
        assert z.shape[0] == past.shape[0]
        assert z.shape[1] == cfg.encoder.latent_channels
        assert z.shape[2:] == past.shape[3:]

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"encoder params: {n_params/1e3:.1f}k")
    print("OK: encoder smoke test passed.")


if __name__ == "__main__":
    main()
