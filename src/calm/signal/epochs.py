"""Cue-aligned epoching for left/right motor-imagery runs."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np

from calm.data.events import (
    LEFT_HAND_LABEL,
    REST_LABEL,
    RIGHT_HAND_LABEL,
    map_left_right_annotations,
)


@dataclass(frozen=True)
class EpochConfig:
    tmin: float = 0.0
    tmax: float = 4.0
    baseline: tuple[float | None, float | None] | None = None


def left_right_epochs_from_raw(
    raw: mne.io.BaseRaw, run: int, config: EpochConfig | None = None, include_rest: bool = False
) -> mne.Epochs:
    """Build epochs from a single run's Raw. By default left/right-only
    (REST excluded), for the supervised task. ``include_rest=True`` also
    includes REST epochs — only valid for unlabeled/self-supervised use,
    since ``epochs_to_arrays`` assumes a two-class left/right label.

    ``raw`` should already be preprocessed (referenced/filtered); mapping
    is applied here so the caller need only pass the raw, unmapped Raw for
    this run.
    """
    config = config or EpochConfig()
    mapped = map_left_right_annotations(raw, run)
    events, event_id = mne.events_from_annotations(mapped, verbose=False)
    wanted_labels = (LEFT_HAND_LABEL, RIGHT_HAND_LABEL, REST_LABEL) if include_rest else (
        LEFT_HAND_LABEL, RIGHT_HAND_LABEL,
    )
    wanted_event_id = {k: v for k, v in event_id.items() if k in wanted_labels}
    return mne.Epochs(
        mapped,
        events,
        event_id=wanted_event_id,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=config.baseline,
        preload=True,
        verbose=False,
    )


def epochs_to_arrays(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with X: (n_epochs, n_channels, n_times), y: 0=left, 1=right."""
    X = epochs.get_data(copy=True)
    label_of_code = {v: k for k, v in epochs.event_id.items()}
    codes = epochs.events[:, -1]
    y = np.array(
        [0 if label_of_code[c] == LEFT_HAND_LABEL else 1 for c in codes], dtype=int
    )
    return X, y
