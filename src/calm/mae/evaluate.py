"""Adapt frozen encoder features to the cross-run evaluator."""

from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LogisticRegression

from calm.eval.cross_run import ModelSpec
from calm.mae.probe import (
    build_encoder_from_checkpoint,
    encoder_feature_fns_from_checkpoint,
    load_checkpoint,
    make_embedding_feature_fn,
)
from calm.representation.stft_frontend import STFTConfig


def pretrained_encoder_spec(checkpoint_path: Path, name: str) -> ModelSpec:
    """Build a named frozen probe for one pretrained checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path)
    model, geometry, normalizer = build_encoder_from_checkpoint(checkpoint, pretrained=True)
    stft_config = STFTConfig(**checkpoint["stft_config"])
    feature_fn = make_embedding_feature_fn(
        model, normalizer, stft_config, geometry.n_time_bins, geometry.bands
    )
    return ModelSpec(name, lambda: LogisticRegression(max_iter=1000), feature_fn=feature_fn)


def encoder_model_specs(checkpoint_path: Path, seed: int = 0) -> list[ModelSpec]:
    fns = encoder_feature_fns_from_checkpoint(checkpoint_path, seed=seed)
    return [
        ModelSpec(
            "random_encoder_logreg",
            lambda: LogisticRegression(max_iter=1000),
            feature_fn=fns["random_encoder"],
        ),
        ModelSpec(
            "pretrained_encoder_logreg",
            lambda: LogisticRegression(max_iter=1000),
            feature_fn=fns["pretrained_encoder"],
        ),
    ]
