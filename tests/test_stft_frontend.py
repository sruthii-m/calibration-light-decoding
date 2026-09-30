from __future__ import annotations

import numpy as np
import pytest

from calm.representation.stft_frontend import STFTConfig, TrainOnlyNormalizer, epochs_to_log_power


def test_epochs_to_log_power_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4, 640)) * 1e-6  # (n_epochs, n_channels, n_times)
    log_power, freqs, times = epochs_to_log_power(X, sfreq=160.0, config=STFTConfig(fmax=40.0))

    assert log_power.shape[:2] == (5, 4)
    assert log_power.shape[2] == len(freqs)
    assert log_power.shape[3] == len(times)
    assert freqs.max() <= 40.0
    assert np.isfinite(log_power).all()


def test_train_only_normalizer_uses_only_fit_data():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((10, 2, 3, 4)) * 2.0 + 5.0
    normalizer = TrainOnlyNormalizer().fit(train)
    mean_before, std_before = normalizer.mean_.copy(), normalizer.std_.copy()

    test = rng.standard_normal((3, 2, 3, 4)) * 100.0 + 1000.0  # wildly different distribution
    transformed_test = normalizer.transform(test)

    np.testing.assert_array_equal(normalizer.mean_, mean_before)
    np.testing.assert_array_equal(normalizer.std_, std_before)
    assert transformed_test.shape == test.shape


def test_train_only_normalizer_zscore_on_train_data():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((200, 1, 1, 1)) * 3.0 + 7.0
    normalizer = TrainOnlyNormalizer()
    transformed = normalizer.fit_transform(train)
    assert transformed.mean() == pytest.approx(0.0, abs=0.15)
    assert transformed.std() == pytest.approx(1.0, abs=0.15)
