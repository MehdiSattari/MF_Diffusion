"""End-to-end smoke test: Sionna CDL data -> encoder -> U-Net generator.

Run on Alvis (needs Sionna + torch):  python -m scripts.smoke_generator
"""

import torch
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset
from mf_csi.models import TemporalEncoder, UNetGenerator


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = CSIStreamDataset(cfg.data, batch_size=4, steps_per_epoch=1)
    loader = DataLoader(ds, batch_size=None)
    enc = TemporalEncoder(cfg.encoder).to(device)
    gen = UNetGenerator(cfg.generator).to(device)

    for batch in loader:
        past = batch["past"].to(device)          # [B, Np, 2, Nt, Nc]
        future = batch["future"].to(device)      # [B, Nf, 2, Nt, Nc]
        z = enc(past)                            # [B, C_z, Nt, Nc]
        h0 = future[:, 0]                        # stand-in CSI frame [B, 2, Nt, Nc]
        B = h0.shape[0]
        r, t = torch.rand(B, device=device), torch.rand(B, device=device)
        u = gen(h0, z, r, t)                     # predicted average velocity
        print("Z", tuple(z.shape), "-> u", tuple(u.shape))
        assert u.shape == h0.shape

    enc_p = sum(p.numel() for p in enc.parameters())
    gen_p = sum(p.numel() for p in gen.parameters())
    print(f"params: encoder {enc_p/1e6:.2f}M | generator {gen_p/1e6:.2f}M")
    print("OK: generator smoke test passed.")


if __name__ == "__main__":
    main()
