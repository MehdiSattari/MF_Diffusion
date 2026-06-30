"""Sionna-based 3GPP CDL channel generation for CSI prediction.

This module is the *only* place that touches TensorFlow / Sionna. It produces
frequency-domain CSI sequences on the fly. To avoid TensorFlow and PyTorch
fighting over GPU memory in the same process, TF is pinned to CPU by default
(``force_cpu=True``); channel generation is cheap enough that this is not a
bottleneck.

The public entry point is :class:`CDLChannelGenerator`. Each call to
``generate(batch_size)`` returns a single NumPy array of shape

    [batch_size, T, 2, Nt, Nc]

where T = num_past + num_future, the size-2 axis stacks (real, imag), Nt is the
number of BS antennas and Nc the number of used subcarriers.

A Sionna version shim is included so the code works with both the legacy
``sionna.channel`` API (<= 0.19) and the newer ``sionna.phy.channel`` API
(>= 1.0).
"""

from __future__ import annotations

import numpy as np

from ..config import DataConfig

_KMH_TO_MS = 1.0 / 3.6


def _configure_tf_cpu(force_cpu: bool) -> None:
    """Hide GPUs from TensorFlow only, so PyTorch keeps full GPU access.

    IMPORTANT: we must NOT set the global ``CUDA_VISIBLE_DEVICES`` env var here.
    TF and PyTorch run in the same process, so clearing that variable would hide
    the GPU from PyTorch as well and break training. Instead we use TF's own
    device-visibility API, which is process-local to TF. (With a ``tensorflow-cpu``
    install this is already a no-op, but it keeps us safe if a GPU-enabled TF is
    ever installed.) Must run before the first TF op."""
    import tensorflow as tf
    if force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # TF already initialised with GPUs visible; nothing we can do now.
            pass


def _import_sionna():
    """Return the Sionna symbols we need, regardless of Sionna version."""
    try:  # Sionna >= 1.0
        from sionna.phy.channel.tr38901 import CDL, AntennaArray
        from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
    except Exception:  # Sionna <= 0.19
        from sionna.channel.tr38901 import CDL, AntennaArray
        from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
    return CDL, AntennaArray, subcarrier_frequencies, cir_to_ofdm_channel


class CDLChannelGenerator:
    """Generates frequency-domain CSI sequences from 3GPP CDL models.

    Each generated batch randomly draws a CDL profile and a delay spread; user
    speed is randomised per sample within ``[min_speed, max_speed]`` via
    Sionna's ``min_speed`` / ``max_speed`` arguments.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        _configure_tf_cpu(cfg.force_sionna_cpu)

        import tensorflow as tf
        self._tf = tf
        (self._CDL, self._AntennaArray,
         self._subcarrier_frequencies, self._cir_to_ofdm_channel) = _import_sionna()

        self._rng = np.random.default_rng(cfg.seed)

        # Base station: 16-element single-polarised ULA (num_cols = Nt).
        self._bs_array = self._AntennaArray(
            num_rows=1,
            num_cols=cfg.num_bs_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=cfg.carrier_frequency,
        )
        # User terminal: single omni antenna.
        self._ut_array = self._AntennaArray(
            num_rows=1,
            num_cols=cfg.num_ut_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=cfg.carrier_frequency,
        )

        # Frequencies of the 16 subcarriers actually used: take indices evenly
        # spaced across the full 300-subcarrier grid.
        all_freqs = self._subcarrier_frequencies(
            cfg.num_subcarriers_total, cfg.subcarrier_spacing
        ).numpy()
        idx = np.linspace(
            0, cfg.num_subcarriers_total - 1, cfg.num_subcarriers_used
        ).round().astype(int)
        self._used_freqs = tf.constant(all_freqs[idx])
        self._sampling_frequency = 1.0 / cfg.ofdm_symbol_duration

    def _make_cdl(self, model: str, delay_spread_s: float):
        return self._CDL(
            model=model,
            delay_spread=delay_spread_s,
            carrier_frequency=self.cfg.carrier_frequency,
            ut_array=self._ut_array,
            bs_array=self._bs_array,
            direction="downlink",
            min_speed=self.cfg.min_speed_kmh * _KMH_TO_MS,
            max_speed=self.cfg.max_speed_kmh * _KMH_TO_MS,
        )

    def generate(self, batch_size: int) -> np.ndarray:
        """Return one batch of CSI sequences, shape [B, T, 2, Nt, Nc] (float32)."""
        cfg = self.cfg
        model = str(self._rng.choice(cfg.cdl_models))
        delay_spread_s = float(
            self._rng.uniform(cfg.min_delay_spread_ns, cfg.max_delay_spread_ns)
        ) * 1e-9
        cdl = self._make_cdl(model, delay_spread_s)

        # Channel impulse response over the sequence length.
        #   a   : [B, num_rx, rx_ant, num_tx, tx_ant, num_paths, T]
        #   tau : path delays
        a, tau = cdl(batch_size, cfg.seq_len, self._sampling_frequency)

        # Frequency response on the used subcarriers:
        #   h : [B, num_rx, rx_ant, num_tx, tx_ant, T, Nc]
        h = self._cir_to_ofdm_channel(self._used_freqs, a, tau, normalize=True)
        h = h.numpy()

        # Collapse the singleton rx/tx-group dims -> [B, rx_ant, tx_ant, T, Nc].
        # Here rx_ant = Nr = 1, tx_ant = Nt = 16.
        h = h[:, 0, :, 0, :, :, :]          # [B, Nr, Nt, T, Nc]
        h = h[:, 0, :, :, :]                # [B, Nt, T, Nc]  (Nr == 1)
        h = np.transpose(h, (0, 2, 1, 3))   # [B, T, Nt, Nc]

        # Split complex -> real/imag channel axis: [B, T, 2, Nt, Nc].
        csi = np.stack([h.real, h.imag], axis=2).astype(np.float32)
        return csi
