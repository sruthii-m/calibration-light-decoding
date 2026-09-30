"""Extract frozen encoder features using the saved pretraining normalizer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from calm.mae.model import MAEConfig, MaskedAutoencoder
from calm.representation.stft_frontend import STFTConfig, TrainOnlyNormalizer, epochs_to_log_power
from calm.representation.tokens import TokenGeometry, pool_to_band_time_tokens


def load_checkpoint(checkpoint_path: Path) -> dict:
    return torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)


def build_encoder_from_checkpoint(
    checkpoint: dict, pretrained: bool, seed: int = 0
) -> tuple[MaskedAutoencoder, TokenGeometry, TrainOnlyNormalizer]:
    geometry_info = checkpoint["geometry"]
    geometry = TokenGeometry.from_channels(
        geometry_info["ch_names"],
        n_time_bins=geometry_info["n_time_bins"],
        bands=tuple(tuple(b) for b in geometry_info["bands"]),
    )
    mae_config = MAEConfig(**checkpoint["mae_config"])

    torch.manual_seed(seed)  # deterministic random-init baseline
    model = MaskedAutoencoder(geometry, mae_config)
    if pretrained:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = TrainOnlyNormalizer()
    normalizer.mean_ = checkpoint["normalizer_mean"]
    normalizer.std_ = checkpoint["normalizer_std"]

    return model, geometry, normalizer


def make_embedding_feature_fn(
    model: MaskedAutoencoder,
    normalizer: TrainOnlyNormalizer,
    stft_config: STFTConfig,
    n_time_bins: int,
    bands: tuple[tuple[float, float], ...],
) -> Callable[[np.ndarray], np.ndarray]:
    def embed(X: np.ndarray) -> np.ndarray:
        log_power, freqs, _times = epochs_to_log_power(X, sfreq=160.0, config=stft_config)
        tokens = pool_to_band_time_tokens(log_power, freqs, bands=bands, n_time_bins=n_time_bins)
        normalized = normalizer.transform(tokens).astype(np.float32)
        with torch.no_grad():
            embeddings = model.embed(torch.from_numpy(normalized))
        return embeddings.numpy()

    return embed


def encoder_feature_fns_from_checkpoint(
    checkpoint_path: Path, seed: int = 0
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Build random and pretrained feature extractors with the same frozen normalizer."""
    checkpoint = load_checkpoint(checkpoint_path)
    stft_config = STFTConfig(**checkpoint["stft_config"])
    n_time_bins = checkpoint["geometry"]["n_time_bins"]
    bands = tuple(tuple(b) for b in checkpoint["geometry"]["bands"])

    fns = {}
    for name, pretrained in (("random_encoder", False), ("pretrained_encoder", True)):
        model, _geometry, normalizer = build_encoder_from_checkpoint(
            checkpoint, pretrained=pretrained, seed=seed
        )
        fns[name] = make_embedding_feature_fn(model, normalizer, stft_config, n_time_bins, bands)
    return fns
