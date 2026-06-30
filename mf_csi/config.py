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
from typing import Tuple
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
    min_speed_kmh: float = 30.0
    max_speed_kmh: float = 120.0
    min_delay_spread_ns: float = 50.0
    max_delay_spread_ns: float = 400.0

    # --- Sequence layout ---
    num_past: int = 30                       # Np  (length of H_past / X)
    num_future: int = 10                     # Nf  (length of H_future / Y)

    # --- Normalization of CSI values fed to the model ---
    # "minmax"  -> per-sample scale to [0, 1]  (paper's choice)
    # "std"     -> per-sample zero-mean / unit-std  (often better for flow models)
    # "none"    -> use Sionna's unit-power normalization as-is
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


@dataclass
class Config:
    """Top-level container. Diffusion / training sections are added in later
    steps."""
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
