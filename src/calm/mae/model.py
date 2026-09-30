"""Transformer masked autoencoder for channel, band, and time tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from calm.mae.masking import build_mask
from calm.representation.tokens import TokenGeometry


@dataclass(frozen=True)
class MAEConfig:
    embed_dim: int = 128
    encoder_depth: int = 4
    encoder_heads: int = 4
    decoder_dim: int = 64
    decoder_depth: int = 2
    decoder_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    mask_ratio: float = 0.6


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TokenPositionalEmbedding(nn.Module):
    """Electrode-position + frequency-band + time-bin embedding, one vector
    per (channel, band, time-bin) token, in the same flatten order as
    ``token_values.reshape(batch, -1)``."""

    def __init__(self, geometry: TokenGeometry, dim: int) -> None:
        super().__init__()
        self.geometry = geometry
        self.channel_proj = nn.Linear(3, dim)
        self.band_embed = nn.Embedding(geometry.n_bands, dim)
        self.time_embed = nn.Embedding(geometry.n_time_bins, dim)
        self.register_buffer(
            "positions", torch.tensor(geometry.positions, dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        ch = self.channel_proj(self.positions)  # (n_channels, dim)
        band = self.band_embed.weight  # (n_bands, dim)
        time = self.time_embed.weight  # (n_time_bins, dim)
        pos = ch[:, None, None, :] + band[None, :, None, :] + time[None, None, :, :]
        return pos.reshape(-1, pos.shape[-1])  # (n_tokens, dim)


class PatchEmbedding(nn.Module):
    def __init__(self, geometry: TokenGeometry, dim: int) -> None:
        super().__init__()
        self.geometry = geometry
        self.value_proj = nn.Linear(1, dim)
        self.pos_embed = TokenPositionalEmbedding(geometry, dim)
        self.missing_embed = nn.Embedding(2, dim)  # 0 = present, 1 = missing channel

    def forward(
        self, token_values: torch.Tensor, missing_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch = token_values.shape[0]
        flat = token_values.reshape(batch, -1, 1)
        value_embed = self.value_proj(flat)
        pos = self.pos_embed().unsqueeze(0)

        if missing_mask is None:
            missing_mask = torch.zeros(
                self.geometry.n_channels, dtype=torch.bool, device=token_values.device
            )
        per_token_missing = (
            missing_mask.view(-1, 1, 1)
            .expand(-1, self.geometry.n_bands, self.geometry.n_time_bins)
            .reshape(-1)
        )
        missing_component = self.missing_embed(per_token_missing.long()).unsqueeze(0)

        return value_embed + pos + missing_component


class MAEDecoder(nn.Module):
    def __init__(
        self,
        geometry: TokenGeometry,
        encoder_dim: int,
        decoder_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed = TokenPositionalEmbedding(geometry, decoder_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, n_heads, mlp_ratio, dropout=0.0) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, 1)

    def forward(self, visible_encoded: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        batch = visible_encoded.shape[0]
        n_tokens = keep_mask.shape[0]
        x_visible = self.decoder_embed(visible_encoded)

        full = self.mask_token.expand(batch, n_tokens, x_visible.shape[-1]).clone()
        full[:, keep_mask, :] = x_visible
        full = full + self.pos_embed().unsqueeze(0)

        for block in self.blocks:
            full = block(full)
        full = self.norm(full)
        return self.head(full).squeeze(-1)  # (batch, n_tokens)


class MaskedAutoencoder(nn.Module):
    def __init__(self, geometry: TokenGeometry, config: MAEConfig | None = None) -> None:
        super().__init__()
        self.geometry = geometry
        self.config = config or MAEConfig()
        config = self.config
        self.patch_embed = PatchEmbedding(geometry, config.embed_dim)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embed_dim, config.encoder_heads, config.mlp_ratio, config.dropout
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(config.embed_dim)
        self.decoder = MAEDecoder(
            geometry,
            config.embed_dim,
            config.decoder_dim,
            config.decoder_depth,
            config.decoder_heads,
            config.mlp_ratio,
        )

    def encode(
        self,
        token_values: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.patch_embed(token_values, missing_mask)
        if keep_mask is not None:
            x = x[:, keep_mask, :]
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_norm(x)

    def forward_loss(
        self,
        token_values: torch.Tensor,
        mask_ratio: float | None = None,
        seed: int = 0,
        mask_strategy: str = "random",
    ) -> torch.Tensor:
        """Mean squared reconstruction error over masked tokens only."""
        mask_ratio = self.config.mask_ratio if mask_ratio is None else mask_ratio
        masked = build_mask(mask_strategy, self.geometry, mask_ratio, seed).to(token_values.device)
        keep_mask = ~masked

        encoded = self.encode(token_values, keep_mask=keep_mask)
        pred = self.decoder(encoded, keep_mask)
        target = token_values.reshape(token_values.shape[0], -1)

        return ((pred[:, masked] - target[:, masked]) ** 2).mean()

    def embed(
        self, token_values: torch.Tensor, missing_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mean-pool frozen encoder outputs with all tokens visible."""
        with torch.no_grad():
            encoded = self.encode(token_values, keep_mask=None, missing_mask=missing_mask)
            return encoded.mean(dim=1)
