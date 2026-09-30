from __future__ import annotations

from calm.preprocessing_sweep import DEFAULT_CANDIDATES


def test_candidate_names_are_unique():
    names = [c.name for c in DEFAULT_CANDIDATES]
    assert len(names) == len(set(names))


def test_default_candidate_matches_baseline_runner_defaults():
    default = next(c for c in DEFAULT_CANDIDATES if c.name.startswith("default"))
    assert default.l_freq == 4.0
    assert default.h_freq == 40.0
    assert default.apply_notch is False
    assert default.apply_reference is True
    assert default.epoch_config.tmin == 0.0
    assert default.epoch_config.tmax == 4.0


def test_each_non_default_candidate_varies_exactly_one_axis():
    default = next(c for c in DEFAULT_CANDIDATES if c.name.startswith("default"))
    for candidate in DEFAULT_CANDIDATES:
        if candidate is default:
            continue
        diffs = [
            candidate.l_freq != default.l_freq,
            candidate.h_freq != default.h_freq,
            candidate.apply_notch != default.apply_notch,
            candidate.apply_reference != default.apply_reference,
            candidate.epoch_config != default.epoch_config,
        ]
        assert sum(diffs) == 1, f"{candidate.name} varies {sum(diffs)} axes, expected 1"
