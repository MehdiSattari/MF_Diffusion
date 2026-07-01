# MeanFlow CSI Prediction

A MeanFlow (one-step generative) variant of diffusion-based CSI prediction,
built around the **DiU** design: a ConvLSTM temporal encoder + a U-Net
generator with **autoregressive** inference. Channel data comes from
**Sionna** 3GPP CDL models, generated on the fly.

This re-frames the diffusion generator of "CSI Prediction Using Diffusion
Models" as a MeanFlow model that learns the *average velocity* field, enabling
1-step (1-NFE) prediction instead of multi-step DDIM sampling.

## Roadmap

- [x] **Step 1 — Data pipeline** (`mf_csi/data/`): Sionna CDL generator +
      PyTorch `(H_past, H_future)` stream.
- [x] **Step 2 — ConvLSTM temporal encoder** (`mf_csi/models/`): history
      `H_past` -> spatial latent `Z [B, C_z, Nt, Nc]`.
- [x] **Step 3 — U-Net generator** (`mf_csi/models/unet.py`): `f_G(h, Z, r, t)`
      -> average velocity `u [B, 2, Nt, Nc]`; JVP-tested for Step 4.
- [x] **Step 4 — MeanFlow objective** (`mf_csi/meanflow.py`): flow interpolant,
      `d/dt u` via forward-mode AD, stop-gradient target, adaptive-weighted loss.
- [x] **Step 5 — Autoregressive 1-step inference** (`mf_csi/inference.py`):
      `Ĥ = ε − u(ε, Z, 0, 1)` rolled out over the horizon + NMSE.
- [ ] **Step 6 — Training loop + SLURM** for Alvis.

## System setup (matches the diffusion paper)

| Parameter            | Value                                  |
|----------------------|----------------------------------------|
| Carrier frequency    | 28 GHz                                 |
| Antennas             | 16 (BS, ULA) x 1 (UT)                  |
| Subcarriers          | 300 total, 16 evenly spaced used       |
| SCS / symbol         | 30 kHz / Tsym ~= 33.3 us               |
| CDL models           | A, B, C, D, E (random per batch)       |
| User speed           | Uniform(30, 120) km/h                  |
| Delay spread         | Uniform(50, 400) ns                    |
| History / horizon    | Np = 30 / Nf = 10                       |
| Sample tensor        | `[T, 2, Nt, Nc] = [40, 2, 16, 16]`     |

## Quick start

```bash
pip install -r requirements.txt

# Logic tests (no Sionna/TF needed):
python -m tests.test_dataset_mock

# Full data smoke test (needs Sionna installed, e.g. on Alvis):
python -m scripts.smoke_data
```

## Notes

- **TF/PyTorch coexistence**: Sionna (TensorFlow) is pinned to CPU by default
  (`DataConfig.force_sionna_cpu=True`) so PyTorch owns the GPU. Channel
  generation is cheap on CPU.
- **Normalization**: per-sample min-max by default (paper uses global min-max).
  Revisit when wiring the flow model in Step 4 — flow/diffusion models often
  prefer a fixed global scale.
- **Sionna version**: the data code works with both `sionna.channel` (<=0.19)
  and `sionna.phy.channel` (>=1.0).
