from __future__ import annotations

import numpy as np

from calm.models.baselines import (
    make_bandpower_logreg_pipeline,
    make_csp_lda_pipeline,
    make_majority_baseline,
    shuffle_labels,
)


def test_csp_fitted_only_on_training_data(synthetic_separable_run_data):
    run_data = synthetic_separable_run_data
    X_train = np.concatenate([run_data[4][0], run_data[8][0]], axis=0)
    y_train = np.concatenate([run_data[4][1], run_data[8][1]], axis=0)

    model_a = make_csp_lda_pipeline(n_components=2)
    model_a.fit(X_train, y_train)
    filters_a = model_a.named_steps["csp"].filters_.copy()

    # Identical training data should produce the same CSP filters.
    model_b = make_csp_lda_pipeline(n_components=2)
    model_b.fit(X_train, y_train)
    filters_b = model_b.named_steps["csp"].filters_.copy()

    np.testing.assert_allclose(filters_a, filters_b)


def test_csp_lda_separates_synthetic_classes(synthetic_separable_run_data):
    run_data = synthetic_separable_run_data
    X_train = np.concatenate([run_data[4][0], run_data[8][0]], axis=0)
    y_train = np.concatenate([run_data[4][1], run_data[8][1]], axis=0)
    X_test, y_test = run_data[12]

    model = make_csp_lda_pipeline(n_components=2)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    assert accuracy > 0.8  # well above chance (0.5) on the separable fixture


def test_shuffle_labels_is_a_permutation_and_deterministic():
    y = np.array([0, 0, 1, 1, 0, 1])
    shuffled = shuffle_labels(y, seed=42)
    assert sorted(shuffled.tolist()) == sorted(y.tolist())
    again = shuffle_labels(y, seed=42)
    np.testing.assert_array_equal(shuffled, again)


def test_majority_baseline_predicts_train_majority_class():
    model = make_majority_baseline()
    X_train = np.zeros((6, 2, 10))
    y_train = np.array([0, 0, 0, 0, 1, 1])  # majority class 0 (4 vs 2)
    model.fit(X_train, y_train)
    preds = model.predict(np.zeros((3, 2, 10)))
    assert set(preds.tolist()) == {0}


def test_bandpower_logreg_pipeline_fits_on_2d_features():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 8))
    y_train = np.array([0, 1] * 10)
    model = make_bandpower_logreg_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(rng.standard_normal((4, 8)))
    assert preds.shape == (4,)
