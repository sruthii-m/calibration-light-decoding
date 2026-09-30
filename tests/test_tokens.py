from __future__ import annotations

import numpy as np

from calm.representation.stft_frontend import epochs_to_log_power
from calm.representation.tokens import DEFAULT_BANDS, TokenGeometry, pool_to_band_time_tokens


def test_token_geometry_shapes():
    ch_names = ["Fc5.", "Fc3.", "C3..", "Cz..", "C4..", "Fc4."]
    geometry = TokenGeometry.from_channels(ch_names, n_time_bins=3)

    assert geometry.n_channels == 6
    assert geometry.n_bands == len(DEFAULT_BANDS)
    assert geometry.n_time_bins == 3
    assert geometry.n_tokens == 6 * len(DEFAULT_BANDS) * 3
    assert geometry.positions.shape == (6, 3)


def test_channel_positions_are_nonzero_for_known_electrodes():
    from calm.representation.tokens import channel_positions

    positions = channel_positions(["C3..", "Cz..", "C4.."])
    assert np.linalg.norm(positions, axis=1).min() > 0.0


def test_pool_to_band_time_tokens_shape_and_band_separation():
    sfreq = 160.0
    n_times = 640
    t = np.arange(n_times) / sfreq
    mu_signal = np.sin(2 * np.pi * 10 * t)  # inside mu band (8-12 Hz)
    X = np.tile(mu_signal, (2, 1, 1))  # (n_epochs=2, n_channels=1, n_times)

    log_power, freqs, _ = epochs_to_log_power(X, sfreq=sfreq)
    tokens = pool_to_band_time_tokens(log_power, freqs, bands=DEFAULT_BANDS, n_time_bins=4)

    assert tokens.shape == (2, 1, len(DEFAULT_BANDS), 4)
    mu_power = tokens[:, :, 0, :].mean()
    beta_power = tokens[:, :, 1, :].mean()
    assert mu_power > beta_power
