"""EEG referencing, offline filtering, and causal filtering utilities."""

from __future__ import annotations

import mne
import numpy as np
from scipy.signal import butter, lfilter


def common_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Return a copy of ``raw`` re-referenced to the common average."""
    referenced = raw.copy()
    referenced.set_eeg_reference("average", projection=False, verbose=False)
    return referenced


def bandpass_filter_offline(
    raw: mne.io.BaseRaw, l_freq: float, h_freq: float
) -> mne.io.BaseRaw:
    """Zero-phase (offline, non-causal) band-pass filter, applied to a copy."""
    filtered = raw.copy()
    filtered.filter(l_freq=l_freq, h_freq=h_freq, phase="zero", verbose=False)
    return filtered


def notch_filter(raw: mne.io.BaseRaw, freqs: list[float] | None = None) -> mne.io.BaseRaw:
    """Notch-filter line noise (default 60 Hz), applied to a copy."""
    freqs = freqs or [60.0]
    filtered = raw.copy()
    filtered.notch_filter(freqs=freqs, verbose=False)
    return filtered


def resample_raw(raw: mne.io.BaseRaw, sfreq: float) -> mne.io.BaseRaw:
    """Resample a copy of ``raw`` to ``sfreq``."""
    resampled = raw.copy()
    resampled.resample(sfreq, verbose=False)
    return resampled


def causal_bandpass(
    data: np.ndarray, sfreq: float, l_freq: float, h_freq: float, order: int = 4
) -> np.ndarray:
    """Apply a forward-only Butterworth filter along the last axis.

    Input shape is (..., n_times). No future samples are used."""
    nyquist = sfreq / 2.0
    b, a = butter(order, [l_freq / nyquist, h_freq / nyquist], btype="band")
    return lfilter(b, a, data, axis=-1)
