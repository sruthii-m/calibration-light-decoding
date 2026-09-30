from __future__ import annotations

import numpy as np

from calm.signal.epochs import EpochConfig, epochs_to_arrays, left_right_epochs_from_raw


def test_epoch_count_and_labels_match_annotations(synthetic_raw_run4):
    # synthetic_raw_run4 codes: T0, T1, T0, T2, T0, T1 -> two T1 (left), one T2 (right)
    epochs = left_right_epochs_from_raw(
        synthetic_raw_run4, run=4, config=EpochConfig(tmin=0.0, tmax=0.5)
    )
    X, y = epochs_to_arrays(epochs)

    assert X.shape[0] == 3
    assert sorted(y.tolist()) == [0, 0, 1]


def test_epoch_alignment_matches_raw_samples(synthetic_raw_run4):
    epochs = left_right_epochs_from_raw(
        synthetic_raw_run4, run=4, config=EpochConfig(tmin=0.0, tmax=0.5)
    )
    raw_data = synthetic_raw_run4.get_data()

    epoch_data_array = epochs.get_data(copy=True)
    for onset_sample, epoch_data in zip(
        epochs.events[:, 0], epoch_data_array, strict=True
    ):
        n_samples = epoch_data.shape[-1]
        expected = raw_data[:, onset_sample : onset_sample + n_samples]
        np.testing.assert_allclose(epoch_data, expected)


def test_epoch_window_is_configurable(synthetic_raw_run4):
    short = left_right_epochs_from_raw(
        synthetic_raw_run4, run=4, config=EpochConfig(tmin=0.0, tmax=0.5)
    )
    long = left_right_epochs_from_raw(
        synthetic_raw_run4, run=4, config=EpochConfig(tmin=0.0, tmax=0.9)
    )
    assert long.get_data(copy=True).shape[-1] > short.get_data(copy=True).shape[-1]


def test_include_rest_adds_rest_epochs(synthetic_raw_run4):
    # synthetic_raw_run4 codes: T0, T1, T0, T2, T0, T1 -> three T0 (rest)
    without_rest = left_right_epochs_from_raw(
        synthetic_raw_run4, run=4, config=EpochConfig(tmin=0.0, tmax=0.5)
    )
    with_rest = left_right_epochs_from_raw(
        synthetic_raw_run4, run=4, config=EpochConfig(tmin=0.0, tmax=0.5), include_rest=True
    )
    assert len(without_rest) == 3
    assert len(with_rest) == 6
    assert "REST" in with_rest.event_id
    assert "REST" not in without_rest.event_id
