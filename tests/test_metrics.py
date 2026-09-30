from __future__ import annotations

import numpy as np

from calm.eval.metrics import bootstrap_ci, compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    metrics = compute_metrics(y_true, y_pred, y_score=y_pred.astype(float))

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[3, 0], [0, 3]]


def test_compute_metrics_chance_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])  # always wrong
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["balanced_accuracy"] == 0.0
    assert metrics["cohen_kappa"] < 0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = y_true.copy()
    flip_idx = rng.choice(200, size=40, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    from sklearn.metrics import balanced_accuracy_score

    point_estimate = balanced_accuracy_score(y_true, y_pred)
    lower, upper = bootstrap_ci(y_true, y_pred, n_boot=500, seed=0)

    assert lower <= point_estimate + 1e-6
    assert upper >= point_estimate - 1e-6
    assert lower <= upper
