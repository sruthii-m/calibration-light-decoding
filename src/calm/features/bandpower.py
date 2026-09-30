"""Mu and beta log-bandpower features."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

MU_BAND = (8.0, 12.0)
BETA_BAND = (13.0, 30.0)
DEFAULT_BANDS = (MU_BAND, BETA_BAND)


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        raise ValueError(
            f"Band {band} has too few frequency bins to integrate; check sfreq/nperseg"
        )
    return trapezoid(psd[..., mask], freqs[mask], axis=-1)


def bandpower_features(
    X: np.ndarray, sfreq: float, bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS
) -> np.ndarray:
    """Return log bandpower per channel: (epochs, channels * bands)."""
    nperseg = min(X.shape[-1], int(sfreq * 2))
    freqs, psd = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)  # (n_epochs, n_channels, n_freqs)
    band_powers = [_band_power(freqs, psd, band) for band in bands]  # each (n_epochs, n_channels)
    stacked = np.concatenate(band_powers, axis=-1)  # (n_epochs, n_channels * len(bands))
    return np.log(np.clip(stacked, a_min=1e-20, a_max=None))
