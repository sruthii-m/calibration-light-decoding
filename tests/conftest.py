"""Synthetic EEG fixtures. No network access required."""

from __future__ import annotations

import mne
import numpy as np
import pytest

SFREQ = 160.0
CH_NAMES = ["Fc5.", "Fc3.", "Fc1.", "Fcz.", "C3..", "C1..", "Cz..", "C2.."]
DURATION_SEC = 20.0


def _make_base_raw(rng: np.random.Generator) -> mne.io.BaseRaw:
    n_channels = len(CH_NAMES)
    n_times = int(SFREQ * DURATION_SEC)
    data = 1e-6 * rng.standard_normal((n_channels, n_times))  # volts, EEG-scale noise
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def _make_annotations(
    codes: list[str], onsets: list[float], duration: float = 4.0
) -> mne.Annotations:
    return mne.Annotations(onset=onsets, duration=[duration] * len(onsets), description=codes)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


@pytest.fixture
def synthetic_raw_run4(rng: np.random.Generator) -> mne.io.BaseRaw:
    """A synthetic raw resembling a PhysioNet left/right run (run 4)."""
    raw = _make_base_raw(rng)
    codes = ["T0", "T1", "T0", "T2", "T0", "T1"]
    onsets = [0.0, 2.0, 7.0, 9.0, 14.0, 16.0]
    raw.set_annotations(_make_annotations(codes, onsets, duration=1.0))
    return raw


@pytest.fixture
def synthetic_raw_run6(rng: np.random.Generator) -> mne.io.BaseRaw:
    """A synthetic raw resembling a both-hands/both-feet run (run 6): out of scope."""
    raw = _make_base_raw(rng)
    codes = ["T0", "T1", "T0", "T2"]
    onsets = [0.0, 2.0, 7.0, 9.0]
    raw.set_annotations(_make_annotations(codes, onsets, duration=1.0))
    return raw


@pytest.fixture
def synthetic_raw_with_flat_channel(rng: np.random.Generator) -> mne.io.BaseRaw:
    """A synthetic raw where one channel is flat (near-zero variance)."""
    raw = _make_base_raw(rng)
    data = raw.get_data()
    data[0, :] = 0.0  # flat-line the first channel
    raw._data[:] = data
    codes = ["T0", "T1", "T0", "T2"]
    onsets = [0.0, 2.0, 7.0, 9.0]
    raw.set_annotations(_make_annotations(codes, onsets, duration=1.0))
    return raw


def _make_separable_epochs(
    rng: np.random.Generator,
    n_trials_per_class: int = 20,
    n_channels: int = 4,
    n_times: int = 640,
    sfreq: float = 160.0,
    left_channel: int = 0,
    right_channel: int = 1,
    signal_freq: float = 10.0,
    signal_amplitude: float = 5e-6,
    noise_amplitude: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) with a channel-specific oscillation marking each class, so
    CSP/bandpower baselines have real class-separable structure to learn
    and a shuffled-label control should collapse to chance."""
    n_trials = 2 * n_trials_per_class
    t = np.arange(n_times) / sfreq
    X = noise_amplitude * rng.standard_normal((n_trials, n_channels, n_times))
    y = np.array([0, 1] * n_trials_per_class)
    rng.shuffle(y)
    burst = signal_amplitude * np.sin(2 * np.pi * signal_freq * t)
    for i, label in enumerate(y):
        ch = left_channel if label == 0 else right_channel
        X[i, ch, :] += burst
    return X, y


@pytest.fixture
def synthetic_separable_run_data(
    rng: np.random.Generator,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-run (X, y) with class-separable spatial signal, for models/eval tests."""
    return {
        run: _make_separable_epochs(rng, n_trials_per_class=15) for run in (4, 8, 12)
    }
