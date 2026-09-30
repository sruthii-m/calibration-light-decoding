"""Within-subject cross-run evaluation with transforms fitted on training runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from calm.eval.metrics import compute_metrics
from calm.models.baselines import shuffle_labels

RunData = dict[int, tuple[np.ndarray, np.ndarray]]  # run -> (X, y)


@dataclass
class ModelSpec:
    name: str
    model_factory: Callable[[], object]
    feature_fn: Callable[[np.ndarray], np.ndarray] | None = None


def _score_of(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def within_subject_cross_run(
    subject: int,
    run_data: RunData,
    spec: ModelSpec,
    shuffled_control: bool = False,
    shuffle_seed: int = 0,
) -> list[dict]:
    """Rotate the held-out run across all runs in ``run_data``; return one
    result dict per fold."""
    runs = sorted(run_data)
    if len(runs) < 2:
        raise ValueError("Need at least two runs to hold one out and train on the rest")

    results = []
    for held_out_run in runs:
        train_runs = [r for r in runs if r != held_out_run]
        # Leakage check: the held-out run must never appear in the training set.
        assert held_out_run not in train_runs

        X_train = np.concatenate([run_data[r][0] for r in train_runs], axis=0)
        y_train = np.concatenate([run_data[r][1] for r in train_runs], axis=0)
        X_test, y_test = run_data[held_out_run]

        if shuffled_control:
            y_train = shuffle_labels(y_train, seed=shuffle_seed + held_out_run)

        if spec.feature_fn is not None:
            X_train = spec.feature_fn(X_train)
            X_test = spec.feature_fn(X_test)

        model = spec.model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = _score_of(model, X_test)

        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics.update(
            subject=subject,
            model=spec.name,
            held_out_run=held_out_run,
            train_runs=train_runs,
            shuffled_control=shuffled_control,
        )
        results.append(metrics)
    return results


def per_subject_results(results: list[dict]) -> pd.DataFrame:
    """Average each subject's metrics across held-out runs."""
    df = pd.DataFrame(results)
    grouped = (
        df.groupby(["model", "shuffled_control", "subject"])
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_cohen_kappa=("cohen_kappa", "mean"),
            n_folds=("held_out_run", "count"),
        )
        .reset_index()
    )
    return grouped


def aggregate_results(results: list[dict]) -> pd.DataFrame:
    """Average subject scores and bootstrap subjects for 95% intervals.

    The three rotations share training data, so folds are not independent samples."""
    df = pd.DataFrame(results)
    rows = []
    for (model, shuffled), group in df.groupby(["model", "shuffled_control"]):
        subject_means = group.groupby("subject")["balanced_accuracy"].mean().to_numpy()
        rng = np.random.default_rng(0)
        n_subjects = len(subject_means)
        boot_means = np.array(
            [
                rng.choice(subject_means, size=n_subjects, replace=True).mean()
                for _ in range(2000)
            ]
        )
        lower, upper = np.quantile(boot_means, [0.025, 0.975])
        rows.append(
            {
                "model": model,
                "shuffled_control": shuffled,
                "n_folds": len(group),
                "n_subjects": n_subjects,
                "mean_balanced_accuracy": float(subject_means.mean()),
                "balanced_accuracy_ci_low": float(lower),
                "balanced_accuracy_ci_high": float(upper),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "mean_cohen_kappa": float(group["cohen_kappa"].mean()),
            }
        )
    return pd.DataFrame(rows)
