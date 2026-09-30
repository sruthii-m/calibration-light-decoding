"""Pool spectral power into tokens and assign electrode coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
from mne.datasets import eegbci

DEFAULT_BANDS: tuple[tuple[float, float], ...] = ((8.0, 12.0), (13.0, 30.0))  # mu, beta


def channel_positions(ch_names: list[str]) -> np.ndarray:
    """(n_channels, 3) standardized 10-10 electrode coordinates, in the same
    order as ``ch_names``. Channels absent from the montage get (0, 0, 0)."""
    info = mne.create_info(ch_names, sfreq=160.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((len(ch_names), 1)), info, verbose=False)
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("colin27_1005")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    ch_pos = raw.get_montage().get_positions()["ch_pos"]

    positions = np.zeros((len(ch_names), 3), dtype=np.float32)
    for i, std_name in enumerate(raw.ch_names):
        pos = ch_pos.get(std_name)
        if pos is not None:
            positions[i] = pos
    return positions


def pool_to_band_time_tokens(
    log_power: np.ndarray,
    freqs: np.ndarray,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
    n_time_bins: int = 4,
) -> np.ndarray:
    """(n_epochs, n_channels, n_freqs, n_times) -> (n_epochs, n_channels, n_bands, n_time_bins),
    mean-pooled within each frequency band and time bin."""
    n_epochs, n_channels, _, n_times = log_power.shape
    band_pooled = np.stack(
        [log_power[:, :, (freqs >= lo) & (freqs <= hi), :].mean(axis=2) for lo, hi in bands],
        axis=2,
    )  # (n_epochs, n_channels, n_bands, n_times)

    time_bin_edges = np.array_split(np.arange(n_times), n_time_bins)
    time_pooled = np.stack(
        [band_pooled[..., edges].mean(axis=-1) for edges in time_bin_edges if len(edges) > 0],
        axis=-1,
    )  # (n_epochs, n_channels, n_bands, n_time_bins)
    return time_pooled


@dataclass(frozen=True)
class TokenGeometry:
    ch_names: list[str]
    positions: np.ndarray  # (n_channels, 3)
    n_bands: int
    n_time_bins: int
    bands: tuple[tuple[float, float], ...] = field(default=DEFAULT_BANDS)

    @property
    def n_channels(self) -> int:
        return len(self.ch_names)

    @property
    def n_tokens(self) -> int:
        return self.n_channels * self.n_bands * self.n_time_bins

    @classmethod
    def from_channels(
        cls,
        ch_names: list[str],
        n_time_bins: int = 4,
        bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
    ) -> TokenGeometry:
        return cls(
            ch_names=list(ch_names),
            positions=channel_positions(ch_names),
            n_bands=len(bands),
            n_time_bins=n_time_bins,
            bands=bands,
        )
