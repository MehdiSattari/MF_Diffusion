# MeanFlow CSI Prediction — Training Pipeline Summary

A complete description of the training pipeline for the one-step generative
(MeanFlow / flow-matching) CSI predictor and its diffusion baseline, as
implemented in the `mf_csi` package. The project reframes the diffusion generator
of *"CSI Prediction Using Diffusion Models"* (Sattari et al.) as a **MeanFlow**
model that learns an *average velocity* field, enabling single-function-evaluation
(1-NFE) prediction. Channel data is generated on the fly with Sionna 3GPP CDL
models; all runs execute on the Alvis (NAISS/Chalmers) cluster.

The current headline model is the **informative-prior (residual-flow) MeanFlow**,
which centers the flow's source distribution on a learned next-frame point
estimate. This recovers strong NMSE without narrowing the prior and, on a
shared-channel evaluation at 20 dB, achieves an average NMSE of **−8.38 dB**,
surpassing the diffusion baseline's **−7.22 dB**.

---

## 1. Task definition

Given a history of `Np = 30` past CSI frames, predict the next `Nf = 10` frames.

A single CSI sample is a real-valued tensor of shape

```
[T, 2, Nt, Nc] = [40, 2, 16, 16]
```

where `T = Np + Nf = 40` time steps, the size-2 axis stacks the real and
imaginary parts of the complex channel, `Nt = 16` is the number of base-station
transmit antennas, and `Nc = 16` is the number of used subcarriers. Batches are
split into history `H_past = X` (`[B, 30, 2, 16, 16]`) and target
`H_future = Y` (`[B, 10, 2, 16, 16]`).

---

## 2. Data generation (Sionna 3GPP CDL)

Channel data is produced on the fly by `mf_csi/data/sionna_cdl.py`
(`CDLChannelGenerator`), the only module that touches TensorFlow / Sionna.

### System parameters

| Parameter | Value |
|---|---|
| Carrier frequency | 28 GHz (mmWave) |
| Subcarrier spacing | 30 kHz |
| Total subcarriers | 300 (25 resource blocks × 12) |
| Used subcarriers `Nc` | 16, evenly spaced across the 300 (via `linspace` + round) |
| OFDM symbol duration `T_sym` | 33.3 µs → time-axis sampling rate `1 / T_sym` |
| BS antenna array | 16-element single-polarized (V) ULA, omni pattern |
| UT antenna array | 1 omni antenna |
| Link direction | Downlink (BS → UT) |
| CDL models | Randomly drawn from {A, B, C, D, E} per batch |

### Channel randomization (paper "uniform" setup)

For each generated batch, `_sample_params` draws:

- a CDL profile uniformly from {A, B, C, D, E};
- a delay spread `U[50, 400] ns` (fixed for the batch);
- user velocity `U[30, 120] km/h`, passed to Sionna as `min_speed` / `max_speed`
  so speed is randomized per sample within the batch.

A second, non-final `matlab_mixture` sampling mode also exists (environment
mixture for delay spread; pedestrian/urban/highway mixture for velocity, up to
~250 km/h) but is **not** used in the paper setup.

### Generation procedure

1. Instantiate a `CDL` object with the sampled profile, delay spread, speeds,
   and the BS/UT arrays.
2. Compute the channel impulse response over the sequence length:
   `a, tau = cdl(batch_size, seq_len, sampling_frequency)`.
3. Convert to the frequency response on the 16 used subcarriers:
   `h = cir_to_ofdm_channel(used_freqs, a, tau, normalize=True)` (Sionna's
   unit-power normalization).
4. Collapse the singleton rx/tx-group dimensions and transpose to `[B, T, Nt, Nc]`.
5. Stack real and imaginary parts on a new channel axis → `[B, T, 2, Nt, Nc]`
   (float32).

A version shim (`_import_sionna`) supports both the legacy `sionna.channel`
(≤ 0.19) and the newer `sionna.phy.channel` (≥ 1.0) APIs.

### TensorFlow / PyTorch GPU coexistence

TensorFlow (Sionna) and PyTorch run in the same process. TF is pinned **off the
GPU** with its own device-visibility API — `tf.config.set_visible_devices([], "GPU")`
— rather than by setting `CUDA_VISIBLE_DEVICES`, which would also hide the GPU
from PyTorch. PyTorch owns the GPU; `tensorflow-cpu` handles channel generation
(cheap enough that CPU is not a bottleneck).

### Dataset wrappers (`mf_csi/data/dataset.py`)

- **`CSIStreamDataset`** — an infinite `IterableDataset` that yields *full
  batches* (Sionna generates a whole batch at once). Used with
  `DataLoader(dataset, batch_size=None)` so PyTorch passes batches through
  unchanged. The generator is built lazily inside the worker process so TF is
  initialized per worker.
- **`make_fixed_eval_set`** — generates a fixed, reproducible list of evaluation
  batches once (for stable validation/test NMSE).
- **`estimate_global_minmax`** — fits a single global `(min, max)` over raw CSI
  (used by the diffusion baseline's global normalization).

Each batch dict carries `past`, `future`, and `stats` (the normalization
parameters needed to invert the transform via `denormalize`).

---

## 3. Normalization

`_normalize` supports several conventions; the choice differs by model because
each is trained (and evaluated) in its own space.

| Mode | Definition | Used by |
|---|---|---|
| `std` | Per-sample zero-mean, unit-std over the full `T` sequence | **MeanFlow (informative prior)** |
| `global_minmax11` | Single global `(a, b)` fit once, mapped to `[−1, 1]` | **Diffusion DiU** (paper convention) |
| `minmax` / `minmax11` | Per-sample scale to `[0, 1]` / `[−1, 1]` | earlier MeanFlow variants |
| `none` | Sionna's unit-power normalization as-is | — |

The switch to **per-sample `std`** for the informative-prior MeanFlow is
deliberate: it makes the data roughly unit-variance, so the flow's source scale
`σ = 1` is **scale-matched** to the data. This is what allows a genuinely wide
prior without the earlier "narrowing" hack.

The diffusion baseline uses `global_minmax11`: a single `(a, b)` is estimated
once over ~4000 samples and reused for train and eval, matching the paper's
dataset-wide min-max and avoiding any per-sample leakage.

> Note: per-sample modes compute their statistics over the *entire* `T` sequence
> (history + future), a mild convention inherited from the original pipeline;
> the global mode avoids this.

### History noise augmentation

To simulate CSI-estimation error, the observed history is corrupted with
**additive, scale-preserving** noise at a per-sample random SNR:

```
X~ = X + σ_n · N,     σ_n chosen per sample so that  signal_power / noise_power = SNR
```

SNR is drawn `U[−20, 20] dB`. Additive noise (rather than the `√ρ · X + N` form)
keeps every frame at its natural scale, which matters in the autoregressive
rollout where observed and predicted frames are mixed.

---

## 4. Network architecture

The MeanFlow model has two sub-networks: a **ConvLSTM temporal encoder** and a
**U-Net generator**.

### 4.1 Temporal encoder (`mf_csi/models/encoder.py`, `conv_lstm.py`)

Maps the history to a spatial conditioning latent `Z` and a next-frame point
estimate `μ`.

```
H_past [B, 30, 2, 16, 16]
  → ConvLSTM (hidden=128, kernel=3, 1 layer)  → top hidden state [B, 128, 16, 16]
  → GroupNorm(1 group) → Dropout(0.2) → Conv2d 3×3 (128→128) → Identity   → Z  [B, 128, 16, 16]
  → (mu head) Conv2d 3×3 (128→2)                                          → μ  [B, 2, 16, 16]
```

- **ConvLSTM cell**: a single 3×3 convolution over `[X_n, Z_{n-1}]` produces all
  four gates `(i, f, o, g)`; `S_n = f·S_{n-1} + i·g`, `Z_n = o·tanh(S_n)`. The
  stack returns the top-layer hidden state at the final time step.
- **`Z`** (128 channels) is the spatial conditioning map concatenated with the
  noisy CSI frame inside the generator.
- **`μ`** (2 channels, linear output) is the informative-prior head:
  `μ ≈ E[Y | history]`. It is returned only when `forward(..., return_mu=True)`,
  so the default call is backward compatible.

### 4.2 U-Net generator (`mf_csi/models/unet.py`)

Predicts the average-velocity field `u` from the noisy CSI frame `h`, the latent
`Z`, and the MeanFlow time pair `(r, t)`.

**Time-pair embedding.** Sinusoidal position embedding (dim 256, inputs scaled by
`time_scale = 1000`) followed by a 2-layer MLP per variable; the `r` and `t`
embeddings are summed.

**Backbone** (input = `concat(h[2ch], Z[128ch]) = 130ch`; spatial 16×16):

| Stage | Operation | Resolution |
|---|---|---|
| in_conv | Conv 3×3 → 32 ch | 16×16 |
| enc1 | 2 × ResBlock(32) → `skip1` | 16×16 |
| down | stride-2 Conv | → 8×8 |
| enc2 | 2 × [ResBlock(→64) + SelfAttn] → `skip2` | 8×8 |
| mid | ResBlock(64) → SelfAttn → ResBlock(64) | 8×8 |
| up1 | concat `skip2`; 3 × [ResBlock(→64) + SelfAttn] | 8×8 |
| up | transpose-Conv 2× | → 16×16 |
| up2 | concat `skip1`; 3 × ResBlock(→32) | 16×16 |
| head | GroupNorm(8) → SiLU → Conv 3×3 (32→2), **zero-init** | 16×16 |

- **ResBlock**: GroupNorm/SiLU/Conv with **FiLM** conditioning — a linear map of
  the time embedding produces `(scale, shift)` applied after the second
  normalization; 1×1 skip when channel counts differ. `norm_groups = 8`,
  `dropout = 0`.
- **SelfAttention2d**: single-head spatial self-attention (GroupNorm, QKV conv,
  softmax) at the 8×8 stage and bottleneck.
- **Zero-initialized output conv**: the network predicts `u ≈ 0` at
  initialization (standard for flow/velocity models).

Key generator hyperparameters: `base_channels = 32`, `ch_mult = 2` (→ 64),
`num_res_blocks = 2`, `time_embed_dim = 256`, `num_heads = 1`.

### 4.3 Diffusion DiU baseline (`mf_csi/models/diu.py`, `diffusion.py`)

A faithful reproduction of the paper's original DiU for like-for-like comparison:

- **`DiUEncoder`**: ConvLSTM (128 hidden) → Conv2d (128→2) + ReLU, emitting a
  2-channel next-frame estimate `Z`.
- **`DiUNet`**: the `diffusers` `UNet2DModel` used directly, with
  `in_channels = 2 (noisy frame) + 2 (Z) = 4`, `out_channels = 2`, a single
  resolution level of width 32, `layers_per_block = 2`, `norm_groups = 1`,
  predicting the **clean frame x0** (`prediction_type = "sample"`).
- **Scheduler**: `DDIMScheduler`, `T = 2000` train timesteps, cosine
  (`squaredcos_cap_v2`) schedule.
- **Sampling**: 20-step DDIM, `eta = 0`, deterministic zero-initialization
  (the sampler trajectory starts from zeros — the paper's "deterministic MMSE"
  behaviour), with a growing history window in the AR rollout.

---

## 5. Training objective

### 5.1 MeanFlow with informative (residual) prior (`mf_csi/meanflow.py`)

**Flow convention.** A straight-line interpolant between the data `Y` (at `t = 0`)
and a source endpoint `S` (at `t = 1`):

```
H^t = (1 − t) · Y + t · S
```

with instantaneous velocity `v = S − Y`.

**Informative prior.** The source is *centered on the point estimate* `μ`:

```
S = stopgrad(μ) + σ · ε ,     ε ~ N(0, I),   σ = source_std = 1.0
```

(Setting `informative_prior = False` recovers the classic `S = σ · ε`.) Because
`S` is centered on a good mean, the flow only has to transport the **residual**
around it, so a single 1-NFE draw already lies near the conditional mean — good
NMSE — while `σ = 1` keeps a genuinely wide, calibrated spread.

**Time-pair sampling** (`sample_r_t`). A logit-normal ("lognorm") sampler draws
two values via `sigmoid(N(P_mean, P_std))` with `P_mean = −0.4`, `P_std = 1.0`;
`t = max`, `r = min`. A fraction `ratio_r_not_equal_t = 0.25` of samples keep
`r ≠ t`; the rest are forced to `r = t` (reducing to flow matching).

**Average velocity and its time-derivative.**

- `u = generator(H^t, Z, r, t)` via a **normal** forward pass — this keeps the
  reverse-mode graph to *both* the generator and the encoder (through `Z`).
- `d/dt u` via a **separate forward-mode-AD** pass (dual tensors) with time
  tangent `(v, 0, 1)` on `(H^t, r, t)`, value only. Two implementation
  subtleties are handled deliberately:
  - `torch.autograd.forward_ad` (not `torch.func.jvp`) is used so that gradients
    still flow to the closed-over network parameters.
  - `u` and `d/dt u` are computed in **separate** passes: fusing them into one
    dual pass drops the reverse-mode edge from `u` back to `Z`, silently zeroing
    the encoder gradient. Since the target is stop-gradient, `d/dt u` needs no
    backward graph.

**Stop-gradient target and loss.**

```
u_tgt   = stopgrad( v − (t − r) · d/dt u )
sq      = per-sample mean of ||u − u_tgt||²
w       = 1 / (stopgrad(sq) + c)^p           # adaptive weight, p = 1, c = 1e-3
flow_loss = mean( w · sq )
```

**Auxiliary point-estimate loss.** The `μ` head is trained *only* by an MSE to
the target (it is stop-gradient'd where it seeds the source), so `μ` learns the
mean and the flow learns the residual transport — a clean separation:

```
mu_loss = MSE(μ, Y)
loss    = flow_loss + mu_loss_weight · mu_loss     # mu_loss_weight = 1.0
```

Reported metrics: `loss`, `flow_loss`, `mse` (raw flow MSE), `mu_mse`, and the
fraction of samples with `r ≠ t`.

### 5.2 Diffusion baseline objective

Single-next-frame denoising: a random history length `t_in ∈ [1, T−1]` is
sampled, the history is SNR-corrupted, `Z = encoder(hist)`, and the U-Net
predicts the clean frame `x0` from the noised frame `x_t`. Loss is **Huber**
(`δ = 0.016`) between predicted and true `x0`.

---

## 6. Inference

### MeanFlow (1-NFE, `mf_csi/inference.py`)

```
Z, μ = encoder(history)
H¹   ~ N(μ, seed_std²)                 # seed_std = source_std = 1.0
u    = generator(H¹, Z, r = 0, t = 1)  # average velocity over [0, 1]
Ŷ    = H¹ − u                          # single function evaluation
```

The predicted frame is appended to the history and the encoder re-run
(**autoregressive** rollout with a growing window) for the full 10-frame horizon.
`num_samples > 1` averages that many 1-NFE draws to approximate the conditional
mean; with the informative prior, `num_samples = 1` is already near-mean.
`step_noise_std` optionally injects per-step stochasticity (0 by default).

### Diffusion (AR DDIM)

Each frame is produced by a 20-step DDIM sampler starting from zeros, conditioned
on `Z`; the window grows each step. This costs ~20 NFE per frame versus the
MeanFlow model's 1 NFE.

### NMSE metric

```
per_step[n] = E_B[ ||H_n − Ĥ_n||² / ||H_n||² ]      # linear
overall     = mean over steps
NMSE(dB)    = 10 · log10(overall)
```

computed in each model's own normalized space.

---

## 7. Training loop and hyperparameters (`scripts/train.py`)

### Optimization

| Setting | Value |
|---|---|
| Optimizer | Adam, `β = (0.9, 0.95)`, `weight_decay = 0` |
| Peak learning rate | `2e-4` |
| LR schedule | Linear warmup (1000 steps) → cosine decay to ~1% of peak |
| Gradient clipping | global-norm clip at `1.0` |
| EMA decay | `0.9999`, warmed up as `min(0.9999, (1+step)/(10+step))` |
| Total steps | 50,000 (headline run) |
| Batch size | 256 (headline run; config default 128) |
| Normalization | `std` (per-sample) |
| Seed | 0 |

### Cadence and evaluation

| Setting | Value |
|---|---|
| `log_every` | 100 steps |
| `eval_every` | 2000 steps |
| `ckpt_every` | 5000 steps |
| Validation set | 256 samples, batch 64, fixed/reproducible |
| Eval inference SNR | 20 dB |

At each evaluation the **EMA weights** are copied into eval-only encoder/generator
clones, autoregressive NMSE is measured on the fixed validation set (per-step and
average, in dB), and the raw (non-EMA) weights are also evaluated for comparison.
`ckpt_best.pt` is saved whenever the EMA average NMSE improves; `ckpt_last.pt` is
saved periodically for resume.

### Loss / MeanFlow objective hyperparameters

| Setting | Value |
|---|---|
| Source scale `σ` (`source_std`) | 1.0 |
| Informative prior | enabled |
| `mu_loss_weight` | 1.0 |
| Time sampler | logit-normal, `P_mean = −0.4`, `P_std = 1.0` |
| `ratio_r_not_equal_t` | 0.25 |
| Adaptive-loss power `p` / eps `c` | 1.0 / 1e-3 |
| History noise-aug SNR | `U[−20, 20] dB` |

### Checkpoint contents

`step`, `best_nmse_db`, `source_std`, encoder/generator state dicts,
`ema_enc` / `ema_gen`, and optimizer state. The saved `source_std` lets
evaluation reconstruct the correct seed scale automatically.

---

## 8. Evaluation and comparison

- **`scripts/evaluate.py`** — per-model figures: NMSE vs prediction step (per
  inference SNR), NMSE vs inference SNR, and ground-truth vs predicted CSI over
  the horizon.
- **`scripts/evaluate_compare.py`** — head-to-head DiU vs MeanFlow. Both models
  are evaluated on **one shared set of Sionna channels** (generated once, then
  normalized in each model's own convention), rolled out autoregressively, and
  overlaid on a single NMSE-vs-step figure. Per-step numbers are also dumped to
  JSON for re-plotting. If a checkpoint lacks a saved `global_ab`, the global
  min-max is re-estimated to reproduce training-time behaviour.
- **Automation** — `slurm/train.sbatch` runs the comparison automatically at the
  end of every training job (`AUTO_COMPARE=1` by default), writing
  `runs/<run>/compare/nmse_vs_step_compare.{png,json}` against the DiU baseline
  (`DIU_CKPT`, overridable).

### Current results (shared-channel eval, EMA weights)

| Model | Condition | step 1 | step 10 | average |
|---|---|---|---|---|
| DiU (diffusion) | 20 dB | −19.98 | −3.01 | −7.22 |
| **MeanFlow (informative prior)** | 20 dB | −17.49 | −4.22 | **−8.38** |
| DiU (diffusion) | clean | −22.16 | −2.26 | −6.71 |
| **MeanFlow (informative prior)** | clean | −18.14 | −3.29 | **−7.68** |

The informative-prior MeanFlow decays more gracefully than DiU: DiU leads at
step 1, but the curves cross around step 3 and MeanFlow ends lower at step 10 and
wins on average — at 1 NFE with a wide `σ = 1` prior.

---

## 9. Infrastructure and execution

- **Cluster**: Alvis (NAISS/Chalmers). Compute account `your-slurm-account`.
- **Environment**: Python 3.11.5 venv; PyTorch owns the GPU, `tensorflow-cpu`
  for Sionna.
- **Required modules** (Sionna on Alvis): `Python/3.11.5-GCCcore-13.2.0` and
  `LLVM/16.0.6-GCCcore-13.2.0`, plus
  `export DRJIT_LIBLLVM_PATH=$(ls $EBROOTLLVM/lib/libLLVM*.so* | head -n1)`.
- **GPUs**: T4 for smoke tests, A40 for real runs.

### Launch commands

```bash
# quick GPU validation (tests + smokes)
sbatch slurm/smoke.sbatch

# real MeanFlow training (auto-generates the comparison plot at the end)
BATCH=256 STEPS=50000 sbatch --gpus-per-node=A40:1 slurm/train.sbatch

# resume
STEPS=100000 RESUME=runs/<run>/ckpt_last.pt OUT=runs/<run> sbatch slurm/train.sbatch

# diffusion baseline
sbatch --gpus-per-node=A40:1 slurm/train_diffusion.sbatch

# standalone comparison plot
DIU=runs/diu_<jobid>/ckpt_best.pt MF=runs/mf_<jobid>/ckpt_best.pt \
    SNRS=20 CLEAN=1 sbatch slurm/eval_compare.sbatch
```

`slurm/train.sbatch` environment overrides: `OUT`, `STEPS`, `BATCH`, `LR`,
`SIGMA` (`--source-std`), `RESUME`, `DIU_CKPT`, `AUTO_COMPARE`.

### CPU-only correctness tests (no Sionna required)

```bash
python -m tests.test_encoder     # ConvLSTM + mu-head shapes / gradients
python -m tests.test_meanflow    # objective: finite loss, gradients to both nets, mu-head grad
python -m tests.test_inference   # autoregressive rollout shapes + NMSE properties
```

---

## 10. Repository layout

```
mf_csi/
  config.py              # all dataclass configs (Data, Encoder, UNet, MeanFlow, Inference, Train, DiU)
  data/
    sionna_cdl.py        # Sionna CDL channel generator (TF-CPU)
    dataset.py           # stream dataset, normalization, fixed eval set
  models/
    conv_lstm.py         # ConvLSTM cell / stack
    encoder.py           # TemporalEncoder (Z + mu head)
    unet.py              # UNetGenerator (FiLM ResBlocks + self-attention)
    diu.py               # diffusion DiU (ConvLSTM + diffusers UNet2DModel)
  meanflow.py            # informative-prior MeanFlow objective (forward-AD JVP)
  inference.py           # 1-NFE autoregressive sampling + NMSE
  diffusion.py           # DDIM schedule, loss, AR sampling (baseline)
  ema.py                 # exponential moving average of weights
scripts/
  train.py               # MeanFlow training loop
  train_diffusion.py     # diffusion baseline training loop
  evaluate.py            # per-model evaluation figures
  evaluate_diffusion.py  # diffusion evaluation (incl. per-velocity sweep)
  evaluate_compare.py    # shared-channel DiU-vs-MeanFlow comparison + plot
  smoke_*.py             # end-to-end smoke tests
slurm/                   # A40/T4 job scripts (train, train_diffusion, eval_compare, smoke)
tests/                   # CPU-only shape/gradient tests
```
