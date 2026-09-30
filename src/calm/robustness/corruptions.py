"""Seeded signal corruptions that return a copy and a record of affected samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorruptionLog:
    kind: str
    affected_channels: list[int]
    affected_samples: tuple[int, int] | None
    params: dict


def random_channel_dropout(
    X: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, CorruptionLog]:
    """Zero out a random subset of channels (all timepoints), on a copy."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    n_channels = X.shape[1]
    rng = np.random.default_rng(seed)
    n_drop = int(round(n_channels * fraction))
    dropped = sorted(rng.choice(n_channels, size=n_drop, replace=False).tolist())

    corrupted = X.copy()
    corrupted[:, dropped, :] = 0.0
    log = CorruptionLog(
        kind="random_channel_dropout",
        affected_channels=dropped,
        affected_samples=None,
        params={"fraction": fraction, "seed": seed},
    )
    return corrupted, log


def contiguous_channel_dropout(
    X: np.ndarray, positions: np.ndarray, n_channels: int, seed: int
) -> tuple[np.ndarray, CorruptionLog]:
    """Zero out a spatially contiguous patch of ``n_channels`` electrodes
    (nearest neighbors of a randomly chosen seed channel, by Euclidean
    distance in 3D electrode-position space)."""
    if not 0 <= n_channels <= X.shape[1]:
        raise ValueError(f"n_channels must be in [0, {X.shape[1]}], got {n_channels}")
    rng = np.random.default_rng(seed)
    seed_channel = int(rng.integers(0, X.shape[1]))
    distances = np.linalg.norm(positions - positions[seed_channel], axis=1)
    dropped = sorted(np.argsort(distances)[:n_channels].tolist())

    corrupted = X.copy()
    corrupted[:, dropped, :] = 0.0
    log = CorruptionLog(
        kind="contiguous_channel_dropout",
        affected_channels=dropped,
        affected_samples=None,
        params={"n_channels": n_channels, "seed": seed, "seed_channel": seed_channel},
    )
    return corrupted, log


def channel_gain_change(
    X: np.ndarray, fraction: float, gain: float, seed: int
) -> tuple[np.ndarray, CorruptionLog]:
    """Scale a random subset of channels by ``gain`` (e.g. 0.1 = attenuated,
    3.0 = amplified), simulating electrode contact/impedance issues."""
    n_channels = X.shape[1]
    rng = np.random.default_rng(seed)
    n_affected = int(round(n_channels * fraction))
    affected = sorted(rng.choice(n_channels, size=n_affected, replace=False).tolist())

    corrupted = X.copy()
    corrupted[:, affected, :] *= gain
    log = CorruptionLog(
        kind="channel_gain_change",
        affected_channels=affected,
        affected_samples=None,
        params={"fraction": fraction, "gain": gain, "seed": seed},
    )
    return corrupted, log


def missing_time_segment(
    X: np.ndarray, segment_frac: float, seed: int
) -> tuple[np.ndarray, CorruptionLog]:
    """Zero out one contiguous time segment (all channels), simulating a
    dropped/corrupted recording interval."""
    if not 0.0 < segment_frac < 1.0:
        raise ValueError(f"segment_frac must be in (0, 1), got {segment_frac}")
    n_times = X.shape[-1]
    seg_len = int(round(n_times * segment_frac))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_times - seg_len + 1))

    corrupted = X.copy()
    corrupted[..., start : start + seg_len] = 0.0
    log = CorruptionLog(
        kind="missing_time_segment",
        affected_channels=list(range(X.shape[1])),
        affected_samples=(start, start + seg_len),
        params={"segment_frac": segment_frac, "seed": seed},
    )
    return corrupted, log


def line_noise_contamination(
    X: np.ndarray, sfreq: float, fraction: float, amplitude: float, freq: float, seed: int
) -> tuple[np.ndarray, CorruptionLog]:
    """Add ``freq`` Hz sinusoidal contamination to a random subset of channels."""
    n_channels = X.shape[1]
    rng = np.random.default_rng(seed)
    n_affected = int(round(n_channels * fraction))
    affected = sorted(rng.choice(n_channels, size=n_affected, replace=False).tolist())

    t = np.arange(X.shape[-1]) / sfreq
    contamination = amplitude * np.sin(2 * np.pi * freq * t)

    corrupted = X.copy()
    corrupted[:, affected, :] += contamination
    log = CorruptionLog(
        kind="line_noise_contamination",
        affected_channels=affected,
        affected_samples=None,
        params={"fraction": fraction, "amplitude": amplitude, "freq": freq, "seed": seed},
    )
    return corrupted, log
