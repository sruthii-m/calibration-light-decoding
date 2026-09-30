from __future__ import annotations

import torch

from calm.mae.masking import random_token_mask
from calm.mae.model import MAEConfig, MaskedAutoencoder
from calm.representation.tokens import TokenGeometry


def _tiny_geometry() -> TokenGeometry:
    return TokenGeometry.from_channels(["C3..", "Cz..", "C4.."], n_time_bins=2)


def _tiny_config() -> MAEConfig:
    return MAEConfig(
        embed_dim=16, encoder_depth=2, encoder_heads=2, decoder_dim=8, decoder_depth=1,
        decoder_heads=2, mlp_ratio=2, dropout=0.0, mask_ratio=0.5,
    )


def test_forward_loss_is_finite_scalar():
    geometry = _tiny_geometry()
    model = MaskedAutoencoder(geometry, _tiny_config())
    token_values = torch.randn(4, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)

    loss = model.forward_loss(token_values, seed=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_masked_tokens_are_hidden_from_encoder():
    """Changing hidden token values must not affect the encoder."""
    geometry = _tiny_geometry()
    model = MaskedAutoencoder(geometry, _tiny_config())
    model.eval()

    token_values = torch.randn(2, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)
    masked = random_token_mask(geometry.n_tokens, mask_ratio=0.5, seed=3)
    keep_mask = ~masked

    encoded_a = model.encode(token_values, keep_mask=keep_mask)

    flat = token_values.reshape(token_values.shape[0], -1).clone()
    flat[:, masked] += 1000.0
    perturbed = flat.reshape(token_values.shape)
    encoded_b = model.encode(perturbed, keep_mask=keep_mask)

    torch.testing.assert_close(encoded_a, encoded_b)


def test_visible_token_perturbation_does_change_encoder_output():
    """The encoder must respond to visible input values."""
    geometry = _tiny_geometry()
    model = MaskedAutoencoder(geometry, _tiny_config())
    model.eval()

    token_values = torch.randn(2, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)
    masked = random_token_mask(geometry.n_tokens, mask_ratio=0.5, seed=3)
    keep_mask = ~masked

    encoded_a = model.encode(token_values, keep_mask=keep_mask)

    flat = token_values.reshape(token_values.shape[0], -1).clone()
    flat[:, keep_mask] += 1000.0
    perturbed = flat.reshape(token_values.shape)
    encoded_b = model.encode(perturbed, keep_mask=keep_mask)

    assert not torch.allclose(encoded_a, encoded_b)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    geometry = _tiny_geometry()
    model = MaskedAutoencoder(geometry, _tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    token_values = torch.randn(8, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        loss = model.forward_loss(token_values, seed=step % 3)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_checkpoint_round_trip_reproduces_embeddings(tmp_path):
    geometry = _tiny_geometry()
    model = MaskedAutoencoder(geometry, _tiny_config())
    model.eval()

    token_values = torch.randn(3, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)
    embedding_before = model.embed(token_values)

    checkpoint_path = tmp_path / "mae.pt"
    torch.save(model.state_dict(), checkpoint_path)

    reloaded = MaskedAutoencoder(geometry, _tiny_config())
    reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    reloaded.eval()
    embedding_after = reloaded.embed(token_values)

    torch.testing.assert_close(embedding_before, embedding_after)


def test_embed_output_shape():
    geometry = _tiny_geometry()
    config = _tiny_config()
    model = MaskedAutoencoder(geometry, config)
    model.eval()
    token_values = torch.randn(5, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)

    embedding = model.embed(token_values)
    assert embedding.shape == (5, config.embed_dim)


def test_embed_does_not_update_parameters():
    geometry = _tiny_geometry()
    model = MaskedAutoencoder(geometry, _tiny_config())
    model.eval()
    token_values = torch.randn(3, geometry.n_channels, geometry.n_bands, geometry.n_time_bins)

    params_before = [p.clone() for p in model.parameters()]
    model.embed(token_values)
    params_after = list(model.parameters())

    for before, after in zip(params_before, params_after, strict=True):
        torch.testing.assert_close(before, after)
