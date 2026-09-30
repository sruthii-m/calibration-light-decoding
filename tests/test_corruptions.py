from __future__ import annotations

import numpy as np
import pytest

from calm.robustness.corruptions import (
    channel_gain_change,
    contiguous_channel_dropout,
    line_noise_contamination,
    missing_time_segment,
    random_channel_dropout,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((5, 8, 100)) + 1.0  # nonzero everywhere


def test_random_channel_dropout_zeros_only_selected_channels(X):
    corrupted, log = random_channel_dropout(X, fraction=0.25, seed=0)
    assert len(log.affected_channels) == 2  # 25% of 8

    for ch in range(8):
        if ch in log.affected_channels:
            assert np.all(corrupted[:, ch, :] == 0.0)
        else:
            np.testing.assert_array_equal(corrupted[:, ch, :], X[:, ch, :])


def test_random_channel_dropout_does_not_mutate_input(X):
    original = X.copy()
    random_channel_dropout(X, fraction=0.5, seed=0)
    np.testing.assert_array_equal(X, original)


def test_random_channel_dropout_is_deterministic(X):
    corrupted_a, log_a = random_channel_dropout(X, fraction=0.5, seed=7)
    corrupted_b, log_b = random_channel_dropout(X, fraction=0.5, seed=7)
    np.testing.assert_array_equal(corrupted_a, corrupted_b)
    assert log_a.affected_channels == log_b.affected_channels


def test_random_channel_dropout_rejects_invalid_fraction(X):
    with pytest.raises(ValueError):
        random_channel_dropout(X, fraction=1.5, seed=0)


def test_contiguous_channel_dropout_drops_nearest_neighbors(X):
    positions = np.array(
        [[i, 0, 0] for i in range(8)], dtype=np.float32
    )  # channels on a line, 0..7
    corrupted, log = contiguous_channel_dropout(X, positions, n_channels=3, seed=1)

    assert len(log.affected_channels) == 3
    dropped = sorted(log.affected_channels)
    assert dropped == list(range(min(dropped), max(dropped) + 1))  # contiguous on the line

    for ch in range(8):
        if ch in log.affected_channels:
            assert np.all(corrupted[:, ch, :] == 0.0)


def test_channel_gain_change_scales_selected_channels(X):
    corrupted, log = channel_gain_change(X, fraction=0.5, gain=0.1, seed=0)
    for ch in range(8):
        if ch in log.affected_channels:
            np.testing.assert_allclose(corrupted[:, ch, :], X[:, ch, :] * 0.1)
        else:
            np.testing.assert_array_equal(corrupted[:, ch, :], X[:, ch, :])


def test_missing_time_segment_zeros_contiguous_block(X):
    corrupted, log = missing_time_segment(X, segment_frac=0.2, seed=0)
    start, end = log.affected_samples
    assert end - start == 20  # 20% of 100

    assert np.all(corrupted[..., start:end] == 0.0)
    before = np.concatenate([corrupted[..., :start], corrupted[..., end:]], axis=-1)
    expected_before = np.concatenate([X[..., :start], X[..., end:]], axis=-1)
    np.testing.assert_array_equal(before, expected_before)


def test_line_noise_contamination_adds_signal_only_to_affected_channels(X):
    corrupted, log = line_noise_contamination(
        X, sfreq=160.0, fraction=0.25, amplitude=5.0, freq=60.0, seed=0
    )
    for ch in range(8):
        if ch not in log.affected_channels:
            np.testing.assert_array_equal(corrupted[:, ch, :], X[:, ch, :])
        else:
            assert not np.allclose(corrupted[:, ch, :], X[:, ch, :])
