from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from calm.mae.evaluate import encoder_model_specs
from calm.mae.model import MAEConfig, MaskedAutoencoder
from calm.mae.probe import build_encoder_from_checkpoint, make_embedding_feature_fn
from calm.representation.stft_frontend import STFTConfig
from calm.representation.tokens import DEFAULT_BANDS, TokenGeometry


def _write_synthetic_checkpoint(tmp_path, ch_names):
    geometry = TokenGeometry.from_channels(ch_names, n_time_bins=2, bands=DEFAULT_BANDS)
    config = MAEConfig(
        embed_dim=8, encoder_depth=1, encoder_heads=2, decoder_dim=4, decoder_depth=1,
        decoder_heads=2, mlp_ratio=2, dropout=0.0, mask_ratio=0.5,
    )
    model = MaskedAutoencoder(geometry, config)
    stft_config = STFTConfig(fmax=40.0)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mae_config": asdict(config),
        "geometry": {
            "ch_names": geometry.ch_names,
            "n_bands": geometry.n_bands,
            "n_time_bins": geometry.n_time_bins,
            "bands": geometry.bands,
        },
        "normalizer_mean": np.zeros((1, len(ch_names), len(DEFAULT_BANDS), 1), dtype=np.float32),
        "normalizer_std": np.ones((1, len(ch_names), len(DEFAULT_BANDS), 1), dtype=np.float32),
        "stft_config": asdict(stft_config),
        "pretraining_subjects": [1, 2],
        "loss_history": [1.0, 0.5],
    }
    path = tmp_path / "mae.pt"
    torch.save(checkpoint, path)
    return path


def test_build_encoder_from_checkpoint_pretrained_matches_saved_weights(tmp_path):
    ch_names = ["C3..", "Cz..", "C4.."]
    path = _write_synthetic_checkpoint(tmp_path, ch_names)
    from calm.mae.probe import load_checkpoint

    checkpoint = load_checkpoint(path)
    model, geometry, normalizer = build_encoder_from_checkpoint(checkpoint, pretrained=True)

    assert geometry.n_channels == 3
    assert normalizer.mean_.shape == (1, 3, len(DEFAULT_BANDS), 1)

    token_values = torch.zeros(2, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)
    embedding = model.embed(token_values)
    assert embedding.shape == (2, checkpoint["mae_config"]["embed_dim"])


def test_random_and_pretrained_encoders_differ(tmp_path):
    ch_names = ["C3..", "Cz..", "C4.."]
    path = _write_synthetic_checkpoint(tmp_path, ch_names)
    from calm.mae.probe import load_checkpoint

    checkpoint = load_checkpoint(path)
    random_model, geometry, _ = build_encoder_from_checkpoint(checkpoint, pretrained=False, seed=0)
    pretrained_model, _, _ = build_encoder_from_checkpoint(checkpoint, pretrained=True, seed=0)

    token_values = torch.randn(2, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)
    random_embed = random_model.embed(token_values)
    pretrained_embed = pretrained_model.embed(token_values)

    assert not torch.allclose(random_embed, pretrained_embed)


def test_embedding_feature_fn_matches_expected_shape(tmp_path):
    ch_names = ["C3..", "Cz..", "C4.."]
    path = _write_synthetic_checkpoint(tmp_path, ch_names)
    from calm.mae.probe import load_checkpoint

    checkpoint = load_checkpoint(path)
    model, geometry, normalizer = build_encoder_from_checkpoint(checkpoint, pretrained=True)
    stft_config = STFTConfig(**checkpoint["stft_config"])

    embed_fn = make_embedding_feature_fn(
        model, normalizer, stft_config, geometry.n_time_bins, geometry.bands
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, geometry.n_channels, 640)) * 1e-6
    features = embed_fn(X)
    assert features.shape == (5, checkpoint["mae_config"]["embed_dim"])


def test_encoder_model_specs_returns_random_and_pretrained(tmp_path):
    ch_names = ["C3..", "Cz..", "C4.."]
    path = _write_synthetic_checkpoint(tmp_path, ch_names)

    specs = encoder_model_specs(path)
    names = {spec.name for spec in specs}
    assert names == {"random_encoder_logreg", "pretrained_encoder_logreg"}
    for spec in specs:
        assert spec.feature_fn is not None
        model = spec.model_factory()
        assert hasattr(model, "fit")
