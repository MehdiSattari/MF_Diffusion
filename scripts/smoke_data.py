"""Smoke test for the Sionna CDL data pipeline.

Run on a machine with Sionna installed (e.g. Alvis):

    python -m scripts.smoke_data

It generates a couple of batches, prints shapes and value ranges, and checks
that past/future split and normalization behave as expected.
"""

from mf_csi.config import DataConfig
from mf_csi.data import CSIStreamDataset
from torch.utils.data import DataLoader


def main():
    cfg = DataConfig()
    print("Sample shape [T,2,Nt,Nc]:", cfg.sample_shape, "| seq_len:", cfg.seq_len)

    ds = CSIStreamDataset(cfg, batch_size=8, steps_per_epoch=2)
    # batch_size=None: the dataset already yields full batches.
    loader = DataLoader(ds, batch_size=None)

    for i, batch in enumerate(loader):
        past, future, stats = batch["past"], batch["future"], batch["stats"]
        print(f"[batch {i}] past {tuple(past.shape)}  future {tuple(future.shape)}  "
              f"norm={stats['mode']}  past[min,max]=[{past.min():.3f},{past.max():.3f}]")
        assert past.shape[1] == cfg.num_past
        assert future.shape[1] == cfg.num_future
        assert past.shape[2:] == (2, cfg.num_bs_ant, cfg.num_subcarriers_used)

    print("OK: data pipeline smoke test passed.")


if __name__ == "__main__":
    main()
