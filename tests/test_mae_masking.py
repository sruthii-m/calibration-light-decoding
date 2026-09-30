from __future__ import annotations

import pytest
import torch

from calm.mae.masking import build_mask, random_token_mask, whole_channel_mask
from calm.representation.tokens import DEFAULT_BANDS, TokenGeometry


def test_mask_has_expected_ratio():
    mask = random_token_mask(n_tokens=100, mask_ratio=0.6, seed=0)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 60


def test_mask_is_deterministic_given_seed():
    mask_a = random_token_mask(n_tokens=50, mask_ratio=0.5, seed=7)
    mask_b = random_token_mask(n_tokens=50, mask_ratio=0.5, seed=7)
    assert torch.equal(mask_a, mask_b)


def test_different_seeds_give_different_masks():
    mask_a = random_token_mask(n_tokens=50, mask_ratio=0.5, seed=1)
    mask_b = random_token_mask(n_tokens=50, mask_ratio=0.5, seed=2)
    assert not torch.equal(mask_a, mask_b)


def test_invalid_mask_ratio_raises():
    with pytest.raises(ValueError):
        random_token_mask(n_tokens=10, mask_ratio=0.0, seed=0)
    with pytest.raises(ValueError):
        random_token_mask(n_tokens=10, mask_ratio=1.0, seed=0)


def _geometry() -> TokenGeometry:
    return TokenGeometry.from_channels(
        ["Fc5.", "Fc3.", "C3..", "Cz..", "C4..", "Fc4."], n_time_bins=3, bands=DEFAULT_BANDS
    )


def test_whole_channel_mask_masks_entire_channels():
    geometry = _geometry()
    mask = whole_channel_mask(geometry, mask_ratio=0.5, seed=0)
    reshaped = mask.reshape(geometry.n_channels, geometry.n_bands, geometry.n_time_bins)

    for ch in range(geometry.n_channels):
        channel_mask = reshaped[ch]
        assert channel_mask.all() or not channel_mask.any(), (
            "a channel must be either fully masked or fully visible"
        )


def test_whole_channel_mask_ratio_matches_channel_fraction():
    geometry = _geometry()
    mask = whole_channel_mask(geometry, mask_ratio=0.5, seed=0)
    reshaped = mask.reshape(geometry.n_channels, geometry.n_bands, geometry.n_time_bins)
    masked_channels = reshaped.all(dim=(1, 2)).sum().item()
    assert masked_channels == 3  # 50% of 6 channels


def test_whole_channel_mask_is_deterministic():
    geometry = _geometry()
    mask_a = whole_channel_mask(geometry, mask_ratio=0.5, seed=3)
    mask_b = whole_channel_mask(geometry, mask_ratio=0.5, seed=3)
    assert torch.equal(mask_a, mask_b)


def test_build_mask_dispatches_correctly():
    geometry = _geometry()
    random_mask = build_mask("random", geometry, mask_ratio=0.5, seed=0)
    channel_mask = build_mask("whole_channel", geometry, mask_ratio=0.5, seed=0)
    assert random_mask.shape == channel_mask.shape == (geometry.n_tokens,)

    with pytest.raises(ValueError, match="Unknown mask strategy"):
        build_mask("bogus", geometry, mask_ratio=0.5, seed=0)
