from __future__ import annotations

import numpy as np
import pytest

from calm.data.events import (
    LEFT_HAND_LABEL,
    REST_LABEL,
    RIGHT_HAND_LABEL,
    get_original_annotation_descriptions,
    left_right_events_from_raw,
    map_left_right_annotations,
    original_annotation_counts,
)


def test_original_annotations_are_preserved_before_mapping(synthetic_raw_run4):
    original = get_original_annotation_descriptions(synthetic_raw_run4)
    assert list(original) == ["T0", "T1", "T0", "T2", "T0", "T1"]

    map_left_right_annotations(synthetic_raw_run4, run=4)

    # The source raw object must be untouched by mapping.
    assert list(synthetic_raw_run4.annotations.description) == ["T0", "T1", "T0", "T2", "T0", "T1"]


def test_mapping_produces_expected_semantic_labels(synthetic_raw_run4):
    mapped = map_left_right_annotations(synthetic_raw_run4, run=4)
    assert list(mapped.annotations.description) == [
        REST_LABEL,
        LEFT_HAND_LABEL,
        REST_LABEL,
        RIGHT_HAND_LABEL,
        REST_LABEL,
        LEFT_HAND_LABEL,
    ]


@pytest.mark.parametrize("invalid_run", [1, 2, 6, 10, 14])
def test_mapping_rejects_non_left_right_runs(synthetic_raw_run4, invalid_run):
    with pytest.raises(ValueError, match="not a left/right"):
        map_left_right_annotations(synthetic_raw_run4, run=invalid_run)


def test_mapping_rejects_unexpected_annotation_codes(synthetic_raw_run4):
    import mne

    raw = synthetic_raw_run4.copy()
    bad_annotations = mne.Annotations(onset=[0.0], duration=[1.0], description=["T3"])
    raw.set_annotations(bad_annotations)
    with pytest.raises(ValueError, match="Unexpected annotation codes"):
        map_left_right_annotations(raw, run=4)


def test_original_annotation_counts(synthetic_raw_run4):
    counts = original_annotation_counts(synthetic_raw_run4)
    assert counts == {"T0": 3, "T1": 2, "T2": 1}


def test_left_right_events_from_raw_returns_original_and_mapped(synthetic_raw_run4):
    events, event_id, original = left_right_events_from_raw(synthetic_raw_run4, run=4)
    assert set(event_id) >= {LEFT_HAND_LABEL, RIGHT_HAND_LABEL, REST_LABEL}
    assert events.shape[0] == 6
    np.testing.assert_array_equal(original, ["T0", "T1", "T0", "T2", "T0", "T1"])
