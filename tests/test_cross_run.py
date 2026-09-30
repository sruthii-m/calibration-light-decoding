from __future__ import annotations

import functools

import numpy as np
import pytest

from calm.eval.cross_run import ModelSpec, aggregate_results, within_subject_cross_run
from calm.features.bandpower import bandpower_features
from calm.models.baselines import make_bandpower_logreg_pipeline, make_csp_lda_pipeline


def test_held_out_run_never_appears_in_training_set(synthetic_separable_run_data):
    spec = ModelSpec("csp_lda", lambda: make_csp_lda_pipeline(n_components=2))
    results = within_subject_cross_run(subject=1, run_data=synthetic_separable_run_data, spec=spec)

    assert len(results) == 3
    for r in results:
        assert r["held_out_run"] not in r["train_runs"]
        assert set(r["train_runs"]) == set(synthetic_separable_run_data) - {r["held_out_run"]}


def test_real_labels_outperform_shuffled_control(synthetic_separable_run_data):
    bandpower_fn = functools.partial(bandpower_features, sfreq=160.0)
    spec = ModelSpec("bandpower_logreg", make_bandpower_logreg_pipeline, feature_fn=bandpower_fn)

    real = within_subject_cross_run(1, synthetic_separable_run_data, spec, shuffled_control=False)
    shuffled = within_subject_cross_run(
        1, synthetic_separable_run_data, spec, shuffled_control=True, shuffle_seed=0
    )

    real_mean = np.mean([r["balanced_accuracy"] for r in real])
    shuffled_mean = np.mean([r["balanced_accuracy"] for r in shuffled])

    assert real_mean > 0.8
    assert shuffled_mean < real_mean
    assert shuffled_mean == pytest.approx(0.5, abs=0.25)


def test_shuffled_control_only_shuffles_training_labels(synthetic_separable_run_data):
    """Shuffling calibration labels must retain the test trial count."""
    spec = ModelSpec("csp_lda", lambda: make_csp_lda_pipeline(n_components=2))
    results = within_subject_cross_run(
        1, synthetic_separable_run_data, spec, shuffled_control=True, shuffle_seed=0
    )
    for r, held_out_run in zip(results, sorted(synthetic_separable_run_data), strict=True):
        _, y_test_true = synthetic_separable_run_data[held_out_run]
        assert r["n_test"] == len(y_test_true)


def test_aggregate_results_groups_by_model_and_control(synthetic_separable_run_data):
    spec = ModelSpec("csp_lda", lambda: make_csp_lda_pipeline(n_components=2))
    real = within_subject_cross_run(
        1, synthetic_separable_run_data, spec, shuffled_control=False
    )
    shuffled = within_subject_cross_run(
        1, synthetic_separable_run_data, spec, shuffled_control=True
    )

    agg = aggregate_results(real + shuffled)

    assert set(agg["shuffled_control"]) == {False, True}
    assert (agg["n_folds"] == 3).all()
    real_row = agg[~agg["shuffled_control"]].iloc[0]
    shuffled_row = agg[agg["shuffled_control"]].iloc[0]
    assert real_row["mean_balanced_accuracy"] > shuffled_row["mean_balanced_accuracy"]
