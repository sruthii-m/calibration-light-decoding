from __future__ import annotations

import numpy as np

from calm.features.bandpower import BETA_BAND, MU_BAND, bandpower_features


def _sine_epochs(freq: float, sfreq: float = 160.0, n_times: int = 640, n_epochs: int = 5):
    t = np.arange(n_times) / sfreq
    signal = np.sin(2 * np.pi * freq * t)
    X = np.tile(signal, (n_epochs, 1, 1))  # (n_epochs, 1 channel, n_times)
    return X


def test_mu_signal_has_higher_mu_than_beta_power():
    X = _sine_epochs(freq=10.0)  # inside MU_BAND (8-12 Hz)
    features = bandpower_features(X, sfreq=160.0, bands=(MU_BAND, BETA_BAND))
    mu_power, beta_power = features[0, 0], features[0, 1]
    assert mu_power > beta_power


def test_beta_signal_has_higher_beta_than_mu_power():
    X = _sine_epochs(freq=20.0)  # inside BETA_BAND (13-30 Hz)
    features = bandpower_features(X, sfreq=160.0, bands=(MU_BAND, BETA_BAND))
    mu_power, beta_power = features[0, 0], features[0, 1]
    assert beta_power > mu_power


def test_output_shape_is_channels_times_bands():
    n_epochs, n_channels = 6, 3
    X = np.random.default_rng(0).standard_normal((n_epochs, n_channels, 640)) * 1e-6
    features = bandpower_features(X, sfreq=160.0, bands=(MU_BAND, BETA_BAND))
    assert features.shape == (n_epochs, n_channels * 2)
