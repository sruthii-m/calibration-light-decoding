"""Majority, bandpower/logistic regression, and CSP/shrinkage-LDA baselines."""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_majority_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def make_bandpower_logreg_pipeline(max_iter: int = 1000, C: float = 1.0) -> Pipeline:
    """Operates on precomputed band-power features, shape (n_epochs, n_features)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, C=C)),
        ]
    )


def make_csp_lda_pipeline(n_components: int = 6) -> Pipeline:
    """Operates on raw epoch arrays, shape (n_epochs, n_channels, n_times)."""
    return Pipeline(
        [
            ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def shuffle_labels(y: np.ndarray, seed: int) -> np.ndarray:
    """Deterministically shuffle labels for the shuffled-label control."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(y, copy=True)
    rng.shuffle(shuffled)
    return shuffled
