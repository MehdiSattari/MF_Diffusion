"""Configuration dataclasses for the MeanFlow CSI prediction project.

All physical/system parameters default to the values reported in the paper
"CSI Prediction Using Diffusion Models" (Sattari et al.) so that the MeanFlow
variant is evaluated under an identical channel setup:

    * Carrier frequency      : 28 GHz (mmWave)
    * Antenna config          : 16 x 1 MIMO (16 BS Tx antennas, 1 UT Rx antenna)
    * Bandwidth               : 25 RBs @ 30 kHz SCS  -> 300 subcarriers
    * Subcarriers used        : 16 evenly spaced across the 300
    * OFDM symbol duration     : Tsym ~= 33.3 us  -> time-axis sampling rate
    * CDL models              : randomly drawn from {A, B, C, D, E}
    * User velocity            : Uniform(30, 120) km/h
    * Delay spread             : Uniform(50, 400) ns
    * History / horizon        : Np = 30 past steps, Nf = 10 future steps

A single CSI sample therefore has shape [T, 2, Nt, Nc] = [T, 2, 16, 16],
where the size-2 axis stacks the real and imaginary parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional
import json


@dataclass
class DataConfig:
    # --- Carrier / OFDM numerology ---
    carrier_frequency: float = 28e9          # Hz
    subcarrier_spacing: float = 30e3         # Hz
    num_subcarriers_total: int = 300         # 25 RBs * 12
    num_subcarriers_used: int = 16           # evenly spaced subset (Nc)
    ofdm_symbol_duration: float = 33.3e-6    # Tsym (s); sampling period on time axis

    # --- Antenna geometry (downlink: BS -> UT) ---
    num_bs_ant: int = 16                     # Nt, ULA columns at the base station
    num_ut_ant: int = 1                      # Nr, single-antenna user

    # --- CDL channel randomization ---
    cdl_models: Tuple[str, ...] = ("A", "B", "C", "D", "E")
    # Parameter sampling per batch:
    #   "uniform"        -> paper setup: velocity U[30,120] km/h, delay spread U[50,400] ns.
    #   "matlab_mixture" -> the non-final MATLAB experiment (environment/mobility mixture,
    #                       velocity ~1-250 km/h). NOT used in the paper.
    param_sampling: str = "uniform"
    min_speed_kmh: float = 30.0
    max_speed_kmh: float = 120.0
    min_delay_spread_ns: float = 50.0
    max_delay_spread_ns: float = 400.0
    # Used only when param_sampling == "matlab_mixture":
    env_probs: Tuple[float, ...] = (0.4, 0.4, 0.2)       # Indoor, UMa, RMa (delay spread)
    mobility_probs: Tuple[float, ...] = (0.3, 0.4, 0.3)  # Pedestrian, Urban, Highway (velocity)

    # --- Sequence layout ---
    num_past: int = 30                       # Np  (length of H_past / X)
    num_future: int = 10                     # Nf  (length of H_future / Y)

    # --- Normalization of CSI values fed to the model ---
    # "minmax"   -> per-sample scale to [0, 1]
    # "minmax11" -> per-sample scale to [-1, 1]  (diffusion convention; DiU uses this)
    # "std"      -> per-sample zero-mean / unit-std
    # "none"     -> use Sionna's unit-power normalization as-is
    normalization: str = "minmax"

    # --- Runtime ---
    force_sionna_cpu: bool = True            # keep TF off the GPU; PyTorch owns it
    seed: int = 0

    @property
    def seq_len(self) -> int:
        """Total time steps generated per sample (history + horizon)."""
        return self.num_past + self.num_future

    @property
    def sample_shape(self) -> Tuple[int, int, int, int]:
        """Shape of one CSI sample: [T, 2, Nt, Nc]."""
        return (self.seq_len, 2, self.num_bs_ant, self.num_subcarriers_used)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class EncoderConfig:
    """ConvLSTM temporal encoder (the DiU `f_TE`).

    Ingests the history H_past [B, Np, in_channels, Nt, Nc] and emits a spatial
    latent Z [B, latent_channels, Nt, Nc] that conditions the U-Net generator.
    Defaults follow the paper: single ConvLSTM layer, 128 hidden channels, 3x3
    kernels, single-group GroupNorm, dropout 0.2.
    """
    in_channels: int = 2                 # real/imag
    hidden_channels: int = 128
    kernel_size: int = 3
    num_layers: int = 1
    latent_channels: int = 128           # channels of Z (conditioning feature map)
    dropout: float = 0.2
    norm_groups: int = 1                 # GroupNorm with 1 group (per paper)
    final_activation: str = "none"       # "none" | "tanh"
    # Informative-prior (residual-flow) MeanFlow: the encoder additionally emits a
    # 2-channel next-frame point estimate mu(Z) ~= E[Y|history], used to CENTER the
    # flow's source distribution (H^1 ~ N(mu, sigma^2)) instead of N(0, sigma^2).
    # The flow then only transports the residual around a good mean, so a single
    # 1-NFE draw already sits near the conditional mean (good NMSE) WITHOUT
    # collapsing the prior width -- the spread stays for the distributional story.
    predict_mu: bool = True              # build the mu head
    mu_channels: int = 2                 # real/imag next-frame point estimate


@dataclass
class UNetConfig:
    """U-Net generator (the DiU backbone `f_G`).

    Predicts the average-velocity field u [B, 2, Nt, Nc] from the noisy CSI frame
    concatenated with the temporal latent Z, conditioned on the MeanFlow time
    pair (r, t). Architecture follows the paper's Appendix B: 32->64 channels,
    a single 2x downsample (16x16 -> 8x8), self-attention at the 8x8 stage and
    bottleneck, skip connections, SiLU + adaptive (FiLM) conditioning.
    """
    in_channels: int = 2                 # noisy CSI real/imag
    cond_channels: int = 128             # Z channels — must match EncoderConfig.latent_channels
    out_channels: int = 2                # predicted u real/imag
    base_channels: int = 32              # channels at full resolution
    ch_mult: int = 2                     # multiplier for the down-sampled stage (-> 64)
    num_res_blocks: int = 2
    time_embed_dim: int = 256
    time_scale: float = 1000.0           # scale t,r in [0,1] before sinusoidal embedding
    num_heads: int = 1
    norm_groups: int = 8                 # GroupNorm groups (divides 32 and 64)
    dropout: float = 0.0
    use_attention: bool = True           # self-attention at the 8x8 stage + bottleneck
                                         # (False -> lightweight "small" backbone)


@dataclass
class MeanFlowConfig:
    """MeanFlow training objective (the paper's Algorithm 2).

    Flow convention: H^t = (1-t) Y + t eps, so t=0 is data (the next CSI frame Y)
    and t=1 is noise. The instantaneous velocity is v = eps - Y. The network
    learns the average velocity u_theta(H^t, Z, r, t); the target is
        u_tgt = v - (t - r) * d/dt u_theta
    with a stop-gradient, and the loss is an adaptively-weighted ||u - sg(u_tgt)||^2.
    """
    # Source (noise endpoint) scale: eps ~ N(0, source_std^2). sigma=1 is standard
    # MeanFlow; a SMALL sigma (0.1-0.3) makes the one-step output nearly
    # deterministic and mean-seeking (low sample variance -> good NMSE, stable AR
    # rollout), mirroring the diffusion DiU's deterministic zero-init DDIM. As
    # sigma -> 0 the objective degenerates into direct regression.
    #
    # With informative_prior=True the source is CENTERED on the encoder's point
    # estimate mu(Z), so sigma is the width of the RESIDUAL prior (on std-normalized
    # data). We keep a genuinely wide sigma=1 -- the good NMSE comes from the
    # informative mean, not from suppressing the variance.
    source_std: float = 1.0

    # Informative (data-dependent) prior: H^1 ~ N(mu(Z), sigma^2), mu trained by an
    # auxiliary MSE-to-Y loss and stop-gradient'd where it seeds the flow, so mu
    # learns ONLY the mean and the flow learns residual transport around it.
    informative_prior: bool = True
    mu_loss_weight: float = 1.0          # weight of the auxiliary MSE(mu, Y) term

    # (r, t) sampling: t >= r, with a fraction forced to r == t (flow-matching).
    time_sampler: str = "lognorm"        # "lognorm" | "uniform"
    lognorm_mean: float = -0.4           # logit-normal mean (P_mean)
    lognorm_std: float = 1.0             # logit-normal std  (P_std)
    ratio_r_not_equal_t: float = 0.25    # fraction of samples with r != t

    # Adaptive loss weight  w = 1 / (||Δ||^2 + c)^p  (stop-grad on w).
    loss_power: float = 1.0              # p
    loss_eps: float = 1e-3               # c

    # Noise-augment the conditioning history (simulates CSI estimation error).
    noise_aug: bool = True
    snr_db_min: float = -20.0
    snr_db_max: float = 20.0


@dataclass
class InferenceConfig:
    """1-step MeanFlow AR sampling (the paper's Algorithm 4)."""
    seed_std: float = 1.0            # H^1 ~ N(mu, seed_std^2) — MUST equal MeanFlowConfig.source_std
    step_noise_std: float = 0.0      # optional per-step stochasticity (0 = deterministic MMSE-style)
    mean_samples: int = 1            # 1-NFE draws averaged per frame (1 = true one-step; low-sigma
                                     # training makes averaging unnecessary)


@dataclass
class TrainConfig:
    """Training loop hyperparameters. Grad clipping + EMA are important here: the
    MeanFlow target drifts during training, so clipping stabilizes updates and the
    EMA weights are what we evaluate."""
    total_steps: int = 50000
    batch_size: int = 128
    lr: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0          # 0 disables
    warmup_steps: int = 1000
    ema_decay: float = 0.9999

    log_every: int = 100
    eval_every: int = 2000
    ckpt_every: int = 5000
    val_samples: int = 256
    val_batch_size: int = 64
    eval_snr_db: Optional[float] = 20.0   # inference SNR for periodic eval (None = clean)

    out_dir: str = "runs/meanflow_diu"
    seed: int = 0


@dataclass
class DiUConfig:
    """Diffusion DiU: ConvLSTM next-frame predictor + diffusers UNet2DModel, DDIM.

    Faithful to the paper's original code: the ConvLSTM emits a 2-channel next-frame
    estimate Z, concatenated onto the noisy diffusion frame; the U-Net predicts the
    clean frame x0 (prediction_type='sample'); cosine schedule, DDIM sampling.
    """
    # ConvLSTM predictor (conditioning encoder)
    lstm_hidden: int = 128
    lstm_kernel: int = 3
    lstm_layers: int = 1
    z_channels: int = 2                  # ConvLSTM output channels (conditioning)
    lstm_activation: str = "relu"        # "relu" | "none"

    # diffusers UNet2DModel. Default is PAPER-FAITHFUL: a single resolution level of
    # width 32, no attention, GroupNorm(1) -- this matches the paper's committed
    # config (config/Config.yml: Unet_block_out_channels=[32], DownBlock2D/UpBlock2D,
    # norm_groups=1), NOT the 32->64+attention that Appendix B's *text* describes.
    # A two-stage attention variant is available for experiments by setting e.g.
    # unet_block_channels=(32, 64), unet_attention=True (needs norm_groups dividing both).
    unet_block_channels: Tuple[int, ...] = (32,)
    unet_attention: bool = False
    unet_layers_per_block: int = 2
    unet_norm_groups: int = 1
    unet_width: int = 32                 # kept for reference (== block_channels[0])

    # diffusion
    num_train_timesteps: int = 2000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "sample"      # predict clean x0
    sampling_steps: int = 20
    ddim_eta: float = 0.0
    huber_delta: float = 0.016
    deterministic_init: bool = True      # start the sampler from zeros (paper behaviour)

    # history noise augmentation (per-sample random SNR)
    train_snr_min: float = -20.0
    train_snr_max: float = 20.0


@dataclass
class RegressionConfig:
    """Joint-horizon ConvLSTM regression baseline (JointRegressor).

    A ConvLSTM temporal encoder over the history feeds a small conv decoder that
    emits ALL Nf future frames in a SINGLE forward pass -- no autoregressive
    rollout, hence NO exposure bias. Trained with plain MSE. This is the honest
    rollout-free NMSE 'ceiling' and a clean ConvLSTM reference for the diffusion
    and MeanFlow generators. Normalized in per-sample 'std' space (same as the
    MeanFlow model) so physical-space NMSE is directly comparable after
    denormalization."""
    in_channels: int = 2
    hidden_channels: int = 128           # ConvLSTM hidden (matches TemporalEncoder)
    kernel_size: int = 3
    num_layers: int = 1
    decoder_channels: int = 128
    num_res_blocks: int = 2
    num_future: int = 10                 # frames emitted jointly (== data.num_future)
    norm_groups: int = 8
    dropout: float = 0.0
    # history noise augmentation (per-sample random SNR), matches the other models
    noise_aug: bool = True
    snr_db_min: float = -20.0
    snr_db_max: float = 20.0


@dataclass
class ARConvLSTMConfig:
    """Standalone autoregressive ConvLSTM baseline (the paper's ConvLSTM predictor).

    ConvLSTM over the history -> a next-frame point estimate; rolled out one frame at
    a time. This is the AR-inference discriminative reference for the AR comparisons
    (MeanFlow / DiU), as opposed to the seq2seq JointRegressor. No output activation
    (CSI is signed). Trained with MSE on the next frame."""
    in_channels: int = 2
    hidden_channels: int = 128
    kernel_size: int = 3
    num_layers: int = 1
    dropout: float = 0.2
    norm_groups: int = 1
    final_activation: str = "none"       # "none" | "tanh"
    noise_aug: bool = True
    snr_db_min: float = -20.0
    snr_db_max: float = 20.0


@dataclass
class Config:
    """Top-level container."""
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    generator: UNetConfig = field(default_factory=UNetConfig)
    meanflow: MeanFlowConfig = field(default_factory=MeanFlowConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    diu: DiUConfig = field(default_factory=DiUConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    ar_convlstm: ARConvLSTMConfig = field(default_factory=ARConvLSTMConfig)
