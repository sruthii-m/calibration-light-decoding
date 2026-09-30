"""Map PhysioNet annotations for left/right imagery runs 4, 8, and 12.

T0 is rest, T1 is left-hand imagery, and T2 is right-hand imagery."""

from __future__ import annotations

from collections import Counter

import mne
import numpy as np

from calm.config import LEFT_RIGHT_RUNS

REST_LABEL = "REST"
LEFT_HAND_LABEL = "LEFT_HAND_IMAGERY"
RIGHT_HAND_LABEL = "RIGHT_HAND_IMAGERY"

LEFT_RIGHT_RUN_LABEL_MAP: dict[str, str] = {
    "T0": REST_LABEL,
    "T1": LEFT_HAND_LABEL,
    "T2": RIGHT_HAND_LABEL,
}


def get_original_annotation_descriptions(raw: mne.io.BaseRaw) -> np.ndarray:
    """Return a copy of the raw, unmapped annotation descriptions (e.g. "T1")."""
    return np.array(raw.annotations.description, copy=True)


def original_annotation_counts(raw: mne.io.BaseRaw) -> dict[str, int]:
    """Counts of each original annotation code, computed before any mapping."""
    return dict(Counter(get_original_annotation_descriptions(raw).tolist()))


def map_left_right_annotations(raw: mne.io.BaseRaw, run: int) -> mne.io.BaseRaw:
    """Return a *copy* of ``raw`` with T0/T1/T2 mapped to semantic labels.

    Only valid for runs 4, 8, 12 (left-hand vs right-hand motor imagery).
    The input ``raw`` is never mutated; its original annotation strings
    remain retrievable via :func:`get_original_annotation_descriptions`.
    """
    if run not in LEFT_RIGHT_RUNS:
        raise ValueError(
            f"Run {run} is not a left/right motor-imagery run {LEFT_RIGHT_RUNS}; "
            "T1/T2 mean something else on runs 6/10/14 (both-hands vs both-feet)."
        )

    original_descriptions = get_original_annotation_descriptions(raw)
    unmapped = set(original_descriptions.tolist()) - set(LEFT_RIGHT_RUN_LABEL_MAP)
    if unmapped:
        raise ValueError(
            f"Unexpected annotation codes {sorted(unmapped)} on run {run}; "
            f"expected a subset of {sorted(LEFT_RIGHT_RUN_LABEL_MAP)}."
        )

    mapped_raw = raw.copy()
    mapped_descriptions = [LEFT_RIGHT_RUN_LABEL_MAP[d] for d in original_descriptions]
    mapped_annotations = mne.Annotations(
        onset=raw.annotations.onset,
        duration=raw.annotations.duration,
        description=mapped_descriptions,
        orig_time=raw.annotations.orig_time,
    )
    mapped_raw.set_annotations(mapped_annotations)
    return mapped_raw


def left_right_events_from_raw(
    raw: mne.io.BaseRaw, run: int
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Build (events, event_id, original_descriptions) for a left/right run.

    ``original_descriptions`` is the untouched T0/T1/T2 array captured
    before mapping, so callers can audit against the source data.
    """
    original_descriptions = get_original_annotation_descriptions(raw)
    mapped_raw = map_left_right_annotations(raw, run)
    events, event_id = mne.events_from_annotations(mapped_raw)
    return events, event_id, original_descriptions
