"""End-to-end diffusion-DiU smoke: short train, then DDIM AR predict + NMSE.

Run on Alvis (needs Sionna + diffusers):  python -m scripts.smoke_diffusion
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mf_csi.config import Config
from mf_csi.data import CSIStreamDataset
from mf_csi.models.diu import DiUEncoder, DiUNet
from mf_csi.diffusion import make_scheduler, diffusion_loss, ddim_ar_predict
from mf_csi.inference import nmse, nmse_db

TRAIN_STEPS = 400


def main():
    cfg = Config()
    cfg.data.normalization = "minmax11"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| train steps:", TRAIN_STEPS)

    enc = DiUEncoder(cfg.diu, in_channels=2).to(device)
    unet = DiUNet(cfg.diu, data_channels=2, image_size=cfg.data.num_subcarriers_used).to(device)
    scheduler = make_scheduler(cfg.diu)
    huber = nn.HuberLoss(delta=cfg.diu.huber_delta)
    opt = torch.optim.Adam(list(enc.parameters()) + list(unet.parameters()), lr=2e-4)

    loader = DataLoader(CSIStreamDataset(cfg.data, batch_size=32, steps_per_epoch=TRAIN_STEPS),
                        batch_size=None)
    enc.train(); unet.train()
    for i, batch in enumerate(loader):
        past, future = batch["past"].to(device), batch["future"].to(device)
        full = torch.cat([past, future], dim=1)
        t_in = int(torch.randint(1, full.shape[1], (1,)).item())
        history, target = full[:, :t_in], full[:, t_in]
        loss, m = diffusion_loss(enc, unet, scheduler, history, target, cfg.diu, huber)
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 100 == 0 or i == TRAIN_STEPS - 1:
            print(f"  step {i:4d} | huber {m['loss'].item():.4f}")

    ds = CSIStreamDataset(cfg.data, batch_size=64, steps_per_epoch=1)
    batch = next(iter(DataLoader(ds, batch_size=None)))
    past, future = batch["past"].to(device), batch["future"].to(device)
    pred = ddim_ar_predict(enc, unet, scheduler, past, cfg.data.num_future, cfg.diu)
    per_step, overall = nmse(pred, future)
    print("\nNMSE per prediction step (dB):")
    for n, v in enumerate(per_step, 1):
        print(f"  step {n:2d}: {nmse_db(v).item():6.2f} dB")
    print(f"average NMSE: {nmse_db(overall).item():.2f} dB")
    print("OK: diffusion-DiU smoke test passed.")


if __name__ == "__main__":
    main()
