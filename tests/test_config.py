from __future__ import annotations

import pytest

from calm.config import AuditConfig, validate_runs, validate_subjects


def test_default_config_is_valid():
    config = AuditConfig()
    assert config.subjects == (1, 2, 3, 4, 5)
    assert config.runs == (4, 8, 12)


def test_validate_subjects_rejects_out_of_range():
    with pytest.raises(ValueError):
        validate_subjects((0,))
    with pytest.raises(ValueError):
        validate_subjects((110,))


def test_validate_subjects_rejects_unbounded_request():
    with pytest.raises(ValueError, match="Refusing to silently process"):
        validate_subjects(tuple(range(1, 70)))


def test_validate_subjects_rejects_empty():
    with pytest.raises(ValueError):
        validate_subjects(())


@pytest.mark.parametrize("invalid_runs", [(6,), (10,), (14,), (4, 6)])
def test_validate_runs_rejects_non_left_right_runs(invalid_runs):
    with pytest.raises(ValueError, match="not left/right"):
        validate_runs(invalid_runs)


def test_validate_runs_accepts_left_right_runs():
    validate_runs((4, 8, 12))
