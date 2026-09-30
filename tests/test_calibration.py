from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calm.calibration import calibration_indices, fit_probe, interval, subject_results


def test_calibration_subsets_are_nested_balanced_and_repeatable():
    y = np.array([0, 1] * 15)
    small = calibration_indices(y, 5, 7, 46, 4)
    large = calibration_indices(y, 10, 7, 46, 4)
    assert set(small) <= set(large)
    np.testing.assert_array_equal(np.bincount(y[small]), [5, 5])
    np.testing.assert_array_equal(np.bincount(y[large]), [10, 10])
    np.testing.assert_array_equal(small, calibration_indices(y, 5, 7, 46, 4))
    assert not np.array_equal(small, calibration_indices(y, 5, 8, 46, 4))
    np.testing.assert_array_equal(calibration_indices(y, None, -1, 46, 4), np.arange(30))


def test_calibration_never_silently_caps_unavailable_budget():
    with pytest.raises(ValueError, match="Only"):
        calibration_indices(np.array([0, 1] * 7), 10, 0, 46, 4)


def test_probe_scaler_uses_selected_calibration_examples_only():
    rng = np.random.default_rng(4)
    pool = rng.normal(size=(30, 6))
    labels = np.array([0, 1] * 15)
    idx = calibration_indices(labels, 5, 0, 46, 4)
    test = rng.normal(size=(15, 6))
    model, _, _ = fit_probe("pretrained_encoder", pool[idx], labels[idx], test, 1.0)
    altered, _, _ = fit_probe("pretrained_encoder", pool[idx], labels[idx], test + 1e5, 1.0)
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, pool[idx].mean(0))
    np.testing.assert_allclose(
        model.named_steps["scaler"].mean_, altered.named_steps["scaler"].mean_
    )
    np.testing.assert_allclose(model.named_steps["clf"].coef_, altered.named_steps["clf"].coef_)
    assert not np.allclose(model.named_steps["scaler"].mean_, pool.mean(0))


def test_aggregation_weights_subjects_and_rotations_not_seed_rows():
    rows = []
    for subject in (46, 47):
        for run, values in [(4, [0.2] * 10), (8, [0.8]), (12, [0.8])]:
            for value in values:
                rows.append(
                    dict(
                        subject=subject,
                        model="test",
                        budget="5",
                        held_out_run=run,
                        balanced_accuracy=value,
                        roc_auc=value,
                        cohen_kappa=value,
                    )
                )
    result = subject_results(pd.DataFrame(rows))
    np.testing.assert_allclose(result.balanced_accuracy, 0.6)
    assert len(result) == 2


def test_paired_interval_uses_differences():
    # The identical paired gain has no between-subject variation even when
    # absolute subject performance differs substantially.
    baseline = np.array([0.2, 0.4, 0.8])
    lo, hi = interval((baseline + 0.05) - baseline, 1000, 0)
    assert lo == pytest.approx(0.05)
    assert hi == pytest.approx(0.05)


def test_artifact_validation_rejects_checkpoint_subject_overlap_and_preprocessing_mismatch():
    import copy

    from calm.calibration import validate_artifacts

    config = dict(subjects=[46], runs=[4, 8, 12], budgets=[5, 10, "all"], sampling_seeds=[0])
    selection = dict(selection_subjects=[37], candidate_params={"name": "test", "l_freq": 4.0})
    common = dict(
        pretraining_subjects=[1],
        preprocessing={"l_freq": 4.0},
        geometry={},
        stft_config={},
        mae_config={},
        normalizer_mean=np.zeros(1),
        normalizer_std=np.ones(1),
    )
    checkpoints = [
        dict(common, mask_strategy="random"),
        dict(common, mask_strategy="whole_channel"),
    ]
    validate_artifacts(config, selection, checkpoints)
    bad = copy.deepcopy(checkpoints)
    bad[0]["pretraining_subjects"] = [46]
    with pytest.raises(ValueError, match="leakage"):
        validate_artifacts(config, selection, bad)
    bad = copy.deepcopy(checkpoints)
    bad[0]["preprocessing"]["l_freq"] = 1.0
    with pytest.raises(ValueError, match="preprocessing differ"):
        validate_artifacts(config, selection, bad)


def test_full_calibration_retains_imbalanced_class_counts():
    y = np.array([0] * 14 + [1] * 16)
    idx = calibration_indices(y, None, -1, 46, 4)
    np.testing.assert_array_equal(np.bincount(y[idx]), [14, 16])
