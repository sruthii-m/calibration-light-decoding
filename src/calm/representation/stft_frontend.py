"""Convert EEG epochs to log-STFT power and normalize with training statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


@dataclass(frozen=True)
class STFTConfig:
    window_sec: float = 0.5
    overlap_frac: float = 0.5
    fmax: float = 40.0


def epochs_to_log_power(
    X: np.ndarray, sfreq: float, config: STFTConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (n_epochs, n_channels, n_times) -> (log_power, freqs, times).

    log_power: (n_epochs, n_channels, n_freqs, n_times_stft), freqs capped at
    ``config.fmax`` since preprocessing already band-passes below that.
    """
    config = config or STFTConfig()
    nperseg = max(int(round(config.window_sec * sfreq)), 8)
    noverlap = int(round(nperseg * config.overlap_frac))
    freqs, times, Zxx = stft(X, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
    power = np.abs(Zxx) ** 2  # (n_epochs, n_channels, n_freqs, n_times_stft)

    freq_mask = freqs <= config.fmax
    power = power[..., freq_mask, :]
    freqs = freqs[freq_mask]

    log_power = np.log(np.clip(power, a_min=1e-20, a_max=None))
    return log_power, freqs, times


class TrainOnlyNormalizer:
    """Per-channel/band normalization using statistics fitted on training data."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, log_power: np.ndarray) -> TrainOnlyNormalizer:
        # log_power: (n_epochs, n_channels, n_freqs, n_times)
        self.mean_ = log_power.mean(axis=(0, 3), keepdims=True)
        self.std_ = log_power.std(axis=(0, 3), keepdims=True)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, log_power: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TrainOnlyNormalizer must be fit before transform")
        return (log_power - self.mean_) / self.std_

    def fit_transform(self, log_power: np.ndarray) -> np.ndarray:
        return self.fit(log_power).transform(log_power)
