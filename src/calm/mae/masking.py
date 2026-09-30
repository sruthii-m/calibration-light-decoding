"""Seeded random-token and whole-channel masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from calm.representation.tokens import TokenGeometry


def random_token_mask(n_tokens: int, mask_ratio: float, seed: int) -> torch.Tensor:
    """Boolean tensor of shape (n_tokens,); True = masked (hidden from encoder,
    reconstruction target). Deterministic given ``seed``."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    rng = np.random.default_rng(seed)
    n_masked = int(round(n_tokens * mask_ratio))
    masked_idx = rng.choice(n_tokens, size=n_masked, replace=False)
    mask = np.zeros(n_tokens, dtype=bool)
    mask[masked_idx] = True
    return torch.from_numpy(mask)


def whole_channel_mask(geometry: TokenGeometry, mask_ratio: float, seed: int) -> torch.Tensor:
    """Hide every band/time token for a random subset of channels."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    rng = np.random.default_rng(seed)
    n_masked_channels = int(round(geometry.n_channels * mask_ratio))
    masked_channels = rng.choice(geometry.n_channels, size=n_masked_channels, replace=False)

    mask = np.zeros((geometry.n_channels, geometry.n_bands, geometry.n_time_bins), dtype=bool)
    mask[masked_channels, :, :] = True
    return torch.from_numpy(mask.reshape(-1))


MASK_STRATEGIES = ("random", "whole_channel")


def build_mask(
    strategy: str, geometry: TokenGeometry, mask_ratio: float, seed: int
) -> torch.Tensor:
    if strategy == "random":
        return random_token_mask(geometry.n_tokens, mask_ratio, seed)
    if strategy == "whole_channel":
        return whole_channel_mask(geometry, mask_ratio, seed)
    raise ValueError(f"Unknown mask strategy '{strategy}'; expected one of {MASK_STRATEGIES}")
