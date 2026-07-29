# MeanFlow vs Diffusion for CSI Prediction — Reliable Results Summary

A consolidated record of the most important and trustworthy findings so far, each
tagged with the architecture that produced it and an explicit note on how much to
trust it. Results are grouped by claim; confounded or pending items are marked so
they are not over-interpreted.

Data throughout: Sionna 3GPP CDL, 28 GHz, 16×1 MIMO, 16 subcarriers, `Np=30`
history / `Nf=10` horizon, velocity U[30,120] km/h, delay spread U[50,400] ns.
Inference SNR = 20 dB unless stated. NMSE lower (more negative dB) is better.

---

## Architectures referenced

Short descriptions so each result below maps to a concrete model.

- **MeanFlow (AR, informative-μ)** — `TemporalEncoder` (ConvLSTM 128-hidden → a
  128-channel conditioning latent `z` **and** a 2-channel next-frame point estimate
  `μ`) + `UNetGenerator` (2D U-Net, 32→64 channels, single-head self-attention, FiLM
  time-pair `(r,t)` conditioning; ~1.46M generator, ~2.2M total). Source is centered
  on μ (`H¹ = μ + σε`, σ=1); **1 NFE** autoregressive sampling. std normalization.
  Checkpoint `mf_6913678`.
- **DiU (AR diffusion, paper reproduction)** — `DiUEncoder` (ConvLSTM 128 → a
  2-channel next-frame estimate) + `diffusers` `UNet2DModel` (single level, width 32,
  **no attention**, ~0.82M total), predict-x0, cosine schedule T=2000, DDIM. global
  min-max [−1,1] normalization. Checkpoint `diu_6793522`. (Matches the paper's
  committed config; runs ~3–4 dB short of the paper due to training/data, not
  architecture.)
- **AR ConvLSTM** — ConvLSTM 128 → 2-channel head, MSE next-frame, autoregressive
  rollout (~0.60M). The discriminative point baseline. Checkpoint `arlstm_6916298`.
- **JointRegressor (seq2seq ConvLSTM)** — ConvLSTM 128 encoder → conv decoder
  emitting **all 10 frames in one pass** (no rollout), MSE (~1.36M). Checkpoint
  `reg_6915230`.
- **Controlled seq2seq trio** — a **shared** encoder-free 2D `UNetGenerator`
  (history stacked as conditioning channels, predicts the whole future block) used
  **identically** by MeanFlow (1 NFE) and diffusion (multi-step); ConvLSTM =
  JointRegressor. global-std normalization, no μ. Small (0.70M) and large (1.45M).

---

## 1. Computational efficiency  ✅ reliable

Benchmark on an A40, full 10-frame horizon, random inputs.
Architectures: MeanFlow = `TemporalEncoder + UNetGenerator`; DiU = `DiUEncoder + diffusers UNet2DModel`.

| Model | Params | Gen evals / horizon | FLOPs | Latency (B=64) | Throughput |
|---|---|---|---|---|---|
| MeanFlow (1 NFE) | 2.21M | 10 | 108 GFLOPs | 388 ms | 165 samples/s |
| DiU (20 NFE) | 0.82M | 200 | 125 GFLOPs | 1171 ms | 55 samples/s |
| DiU (3 NFE) | 0.82M | 30 | 109 GFLOPs | 482 ms | — |

**Findings:** DiU is **latency-bound** (many sequential kernel launches), not
FLOP-bound — its FLOPs barely change from 1→20 NFE but latency triples. So
MeanFlow's speed edge is real but **operating-point dependent**: ~3× faster than
20-step DiU, but only **~1.2–1.5× faster than 3-step DiU** (the paper's efficient
point). MeanFlow uses more parameters (bigger backbone) but fewer, cheaper passes.

---

## 2. Controlled MeanFlow-vs-diffusion NMSE (shared backbone, no μ)  ✅ reliable (relative ordering)

The fair objective comparison: identical shared U-Net backbone, same
normalization/training, encoder-free seq2seq, **no μ prior**. Own-eval NMSE, 20 dB.

| Objective (large, 1.45M) | avg NMSE | step 1 | step 10 |
|---|---|---|---|
| Regression (ConvLSTM) | −11.97 dB | −17.73 | −8.84 |
| **Diffusion** (multi-step) | **−4.28 dB** | −6.81 | −3.22 |
| **MeanFlow** (1 NFE) | **−2.34 dB** | −7.90 | −0.13 |

(Small backbone confirms the ordering: Diffusion −3.99 vs MeanFlow −1.00.)

**Findings:** On equal footing without μ, **diffusion beats MeanFlow on NMSE**, and
regression dominates both. MeanFlow's 1-NFE step cannot reach the conditional mean in
one shot (step-10 ≈ 0 dB); diffusion's multi-step refinement gets closer. **Caveat:**
absolute NMSE is weak here because seq2seq (predict all frames from history alone) is
hard at these capacities — trust the **ordering**, not the magnitudes.

---

## 3. Uncertainty & downstream (AR: informative-μ MeanFlow vs stochastic DiU vs AR ConvLSTM)  ⚠️ confounded — see caveat

Physical-space, 20 dB, K=30 samples, target outage 10%. Architectures as in the
reference list (each model its own; **not** a shared backbone).

| Model | avg NMSE | CRPS ↓ | coverage@90% | spread/skill | goodput@10%-outage | achieved outage |
|---|---|---|---|---|---|---|
| **MeanFlow (μ)** | −10.80 dB | 0.100 | 0.72 | 0.61 | 8.58 b/s/Hz | 0.18 |
| DiU (stochastic) | −7.64 dB | 0.165 | 0.36 | 0.28 | 4.89 b/s/Hz | 0.53 |
| AR ConvLSTM (point) | −9.05 dB | — | 0.00 | — | 4.43 b/s/Hz | 0.58 |

Spectral efficiency was **saturated** (~10.5 b/s/Hz for all vs 10.64 perfect) — it
does **not** discriminate at 20 dB.

**Reliable takeaways (robust regardless of the confound):**
- A generative model can provide **calibrated uncertainty a point predictor
  structurally cannot** (coverage 0.72 vs 0.00; outage 0.18 vs 0.58).
- **Calibration quality matters:** a poorly-calibrated generative model (DiU here,
  coverage 0.36) is nearly as bad as a point model on the outage metric — being
  generative is not enough.
- The **outage-constrained rate** metric separates the models where SE cannot.

**Caveat (do not over-read as "MeanFlow beats diffusion"):** this DiU is the weak
reproduction, forced into stochastic sampling (η=1, random init) to yield an
ensemble, and — critically — has **no μ prior** while MeanFlow does. So this compares
*informative-μ MeanFlow* against a *weak, μ-less, stochastically-sampled DiU*. The
unconfounded answer comes from the 2×2 μ-ablation (Section 6, pending).

---

## 4. The μ decomposition (μ-only vs full MeanFlow)  ✅ reliable

Same MeanFlow model; μ-only uses the encoder's point estimate alone (no flow).
Physical space, 20 dB.

| | step 1 | step 10 | avg |
|---|---|---|---|
| μ-only (regression head) | −20.8 | −3.06 | −7.93 |
| Full MeanFlow (μ + flow) | −18.0 | −4.43 | −8.71 |

**Findings:** μ carries **most** of the NMSE; the generative flow adds ~0.8 dB
average (more when clean), concentrated at **long horizon** — the point estimate is
best at step 1 but decays fastest, and the flow's value is keeping the rollout
stable. This quantifies exactly how much of MeanFlow's accuracy is the (deterministic)
point estimate vs the generative residual.

---

## 5. Calibration is tunable via rollout noise  ✅ reliable

Same MeanFlow model, varying the per-step rollout noise. 20 dB, K=30.

| step-noise | coverage@90% | spread/skill | avg NMSE | CRPS |
|---|---|---|---|---|
| 0.00 | 0.68 | 0.52 | −10.56 dB | 0.1025 |
| 0.05 | 0.72 | 0.61 | −10.38 dB | 0.1020 |

**Findings:** MeanFlow is somewhat overconfident (coverage below nominal), worsening
with horizon; injecting per-step noise improves calibration at a small (~0.2 dB) NMSE
cost. Importantly, MeanFlow **has this knob** to trade accuracy for calibration; a
point predictor has none. (Its calibration is genuine — spread tracked skill at short
horizon — so μ did **not** make the process deterministic in practice.)

---

## 6. Pending — the experiment that concludes the study

The clean answer to "does MeanFlow beat diffusion on NMSE / reliability / compute" is
the **2×2 μ-ablation** (running): {MeanFlow, diffusion} × {μ, no-μ}, all on the
**identical `TemporalEncoder + UNetGenerator`**, AR inference, MeanFlow 1 NFE ≤
diffusion 3 NFE, evaluated on NMSE **and** reliability **and** compute. Reuses
`mf_6913678` for the MeanFlow+μ cell. This is the only comparison with the encoder,
backbone, normalization, and training all shared and μ properly controlled.

Also planned: a **wide-posterior regime** (low SNR 5–10 dB / longer horizon), where
the conditional future is genuinely uncertain — the setting in which the generative
residual, μ or not, is doing real work rather than dressing up a near-deterministic
forecast.

---

## 7. Resolved downstream picture — 2×2 μ-ablation (CRPS-rate + global-rate goodput)  ✅ reliable

The unconfounded downstream evaluation on the shared-backbone 2×2 (identical encoder,
backbone, normalization, training; MeanFlow 1 NFE, diffusion 3 NFE). Physical space,
20 dB, K=30. Rate `c = log₂(1 + SNR·Σₐ|h|²)`; **CRPS-rate** scores the whole predicted
rate distribution (proper, un-gameable); goodput selects a rate at 10% target outage.

| Model | NMSE | cov@90 | **CRPS-rate ↓** | global goodput (Case 1) | per-coef. goodput (Case 2) | rate std |
|---|---|---|---|---|---|---|
| **MeanFlow+μ** | −9.82 | 0.71 | **0.163** | **9.73** | 8.74 | 0.39 |
| Diffusion+μ | −10.61 | 0.19 | 0.172 | 9.60 | 6.48 | 0.40 |
| ConvLSTM | −8.55 | — | 0.207 | 8.94 | 4.42 | 0.35 |
| MeanFlow−μ | −6.29 | 0.54 | 0.311 | 8.12 | 4.40 | 0.38 |
| Diffusion−μ | −4.35 | 0.10 | **0.739** | 9.45 | *9.44* | 0.48 |

**The Diffusion−μ goodput anomaly is resolved — it was a metric artifact, not a real gain.**

1. **It's variance, not bias.** Diffusion−μ has the *lowest* mean selected rate (9.83,
   most conservative) but the *highest* spread (std 0.48). Under a per-coefficient rate +
   **aggregate** outage rule (Case 2), that spread lets it allocate the outage budget
   unevenly and spuriously report the highest goodput (9.44) — despite the worst NMSE and
   calibration by a wide margin.
2. **A single global rate removes the artifact (Case 1).** Ordering becomes sensible:
   MeanFlow+μ 9.73 > Diffusion+μ 9.60 > Diffusion−μ 9.45 > ConvLSTM 8.94 > MeanFlow−μ 8.12.
   Diffusion−μ drops from "best" to mid-pack.
3. **CRPS-rate (proper score) ranks it correctly last** (0.739, 4.5× worse than
   MeanFlow+μ's 0.163) — consistent with its NMSE/calibration, opposite to its gamed goodput.

**Why even Case-1 goodput discriminates weakly at 20 dB (don't over-read 9.45 ≈ 9.73):**
the rate is `log₂` of antenna-**summed** power (element errors partially cancel, then the
log compresses residual error), goodput deliberately picks a conservative 10% quantile
(a lower R buys back reliability along a nearly flat trade-off), and the true rate is
**saturated** near the 10.6 b/s/Hz ceiling. Same saturation that flattened spectral
efficiency. → **Report CRPS-rate as the primary downstream metric**; use goodput only to
confirm the outage constraint is met, and read it at **low SNR (5–10 dB)** if an
operational throughput number that actually separates the models is needed.

**Also confirmed (honest caveat):** the channel-level calibration gap (coverage 0.73 vs
0.18) is *larger* than the rate-level CRPS gap (0.163 vs 0.172), because the maximum-ratio
rate aggregates over antennas and blunts per-element mis-calibration — diffusion+μ's good
mean keeps its rate prediction competitive despite poor per-element calibration.

---

## Bottom line so far

1. **Regression owns NMSE** at these horizons/SNR (its optimum is the conditional
   mean); no generative model beats it on the point metric, by construction.
2. **On a fair footing without μ, diffusion ≥ MeanFlow on NMSE** — 1-NFE is a real
   handicap for point accuracy; the informative μ prior is what makes MeanFlow
   competitive.
3. **Generative models add calibrated uncertainty a point model cannot**, and only a
   *well-calibrated* one converts it into downstream (outage) value — MeanFlow(μ) did,
   the weak DiU did not, but that comparison is confounded.
4. **MeanFlow's compute advantage is modest** (~1.2–1.5×) once diffusion runs at its
   3-step operating point.
5. **Downstream, calibration carries over:** MeanFlow+μ has the best CRPS-rate; the
   Diffusion−μ "best goodput" was a per-coefficient-variance artifact that a single
   global rate removes. CRPS-rate is the metric to report; goodput saturates at 20 dB.

The 2×2 ablation + a wide-posterior evaluation will settle whether the honest framing
is "informative-prior MeanFlow: comparable accuracy, slightly cheaper, better-
calibrated at 1 NFE," or "at matched compute, few-step diffusion is at least as good."
