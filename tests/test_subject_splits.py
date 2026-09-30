from __future__ import annotations

import json

import pandas as pd
import pytest

from calm.config import (
    DEV_SUBJECTS,
    TEST_SUBJECTS,
    VALIDATION_SUBJECTS,
    assert_disjoint_subject_splits,
)


def test_splits_are_pairwise_disjoint():
    assert_disjoint_subject_splits()  # must not raise
    dev, val, test = set(DEV_SUBJECTS), set(VALIDATION_SUBJECTS), set(TEST_SUBJECTS)
    assert dev.isdisjoint(val)
    assert dev.isdisjoint(test)
    assert val.isdisjoint(test)


def test_splits_cover_subjects_1_to_60_exactly_once():
    union = sorted(set(DEV_SUBJECTS) | set(VALIDATION_SUBJECTS) | set(TEST_SUBJECTS))
    assert union == list(range(1, 61))
    total = len(DEV_SUBJECTS) + len(VALIDATION_SUBJECTS) + len(TEST_SUBJECTS)
    assert total == 60  # no duplicates across splits


def test_expected_split_sizes():
    assert DEV_SUBJECTS == tuple(range(1, 37))
    assert VALIDATION_SUBJECTS == tuple(range(37, 46))
    assert TEST_SUBJECTS == tuple(range(46, 61))


def test_assert_disjoint_subject_splits_detects_overlap(monkeypatch):
    import calm.config as config

    monkeypatch.setattr(config, "DEV_SUBJECTS", (1, 2, 37))  # overlaps VALIDATION_SUBJECTS
    with pytest.raises(ValueError, match="not disjoint"):
        config.assert_disjoint_subject_splits()


def test_select_preprocessing_rejects_dev_subjects(monkeypatch):
    from calm import preprocessing_selection

    called = {"sweep": False}

    def fake_sweep(*args, **kwargs):
        called["sweep"] = True
        raise AssertionError("sweep must not run when subjects are invalid")

    monkeypatch.setattr(preprocessing_selection, "run_preprocessing_sweep", fake_sweep)

    with pytest.raises(ValueError, match="only validation subjects"):
        preprocessing_selection.select_preprocessing_candidate(subjects=(1, 2, 3))
    assert called["sweep"] is False  # rejected before touching any data


def test_select_preprocessing_rejects_test_subjects(monkeypatch):
    from calm import preprocessing_selection

    def fake_sweep(*args, **kwargs):
        raise AssertionError("sweep must not run when subjects are invalid")

    monkeypatch.setattr(preprocessing_selection, "run_preprocessing_sweep", fake_sweep)

    with pytest.raises(ValueError, match="only validation subjects"):
        preprocessing_selection.select_preprocessing_candidate(subjects=(46, 47))


def test_select_preprocessing_rejects_mixed_valid_and_invalid_subjects(monkeypatch):
    from calm import preprocessing_selection

    def fake_sweep(*args, **kwargs):
        raise AssertionError("sweep must not run when subjects are invalid")

    monkeypatch.setattr(preprocessing_selection, "run_preprocessing_sweep", fake_sweep)

    with pytest.raises(ValueError, match="only validation subjects"):
        preprocessing_selection.select_preprocessing_candidate(subjects=(37, 38, 46))


def test_select_preprocessing_accepts_validation_subjects(monkeypatch, tmp_path):
    from calm import preprocessing_selection

    fake_sweep_df = pd.DataFrame(
        [
            {
                "preprocessing_candidate": "default_car_4to40hz_no_notch_cue0s",
                "model": "csp_lda",
                "shuffled_control": False,
                "mean_balanced_accuracy": 0.6,
            },
            {
                "preprocessing_candidate": "default_car_4to40hz_no_notch_cue0s",
                "model": "bandpower_logreg",
                "shuffled_control": False,
                "mean_balanced_accuracy": 0.62,
            },
            {
                "preprocessing_candidate": "with_60hz_notch",
                "model": "csp_lda",
                "shuffled_control": False,
                "mean_balanced_accuracy": 0.55,
            },
            {
                "preprocessing_candidate": "with_60hz_notch",
                "model": "bandpower_logreg",
                "shuffled_control": False,
                "mean_balanced_accuracy": 0.56,
            },
        ]
    )
    monkeypatch.setattr(
        preprocessing_selection, "run_preprocessing_sweep", lambda **kwargs: fake_sweep_df
    )

    output_path = tmp_path / "selection.json"
    result = preprocessing_selection.select_preprocessing_candidate(
        subjects=(37, 38, 39), output_path=output_path
    )

    assert result["chosen_candidate"] == "default_car_4to40hz_no_notch_cue0s"
    assert result["selection_subjects"] == [37, 38, 39]
    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved["chosen_candidate"] == "default_car_4to40hz_no_notch_cue0s"


def test_evaluate_held_out_rejects_dev_and_validation_subjects(monkeypatch, tmp_path):
    from calm import held_out_evaluation

    def fake_run_baseline_evaluation(*args, **kwargs):
        raise AssertionError("must not run when subjects are invalid")

    monkeypatch.setattr(
        held_out_evaluation, "run_baseline_evaluation", fake_run_baseline_evaluation
    )

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "chosen_candidate": "default",
                "selection_subjects": [37, 38, 39],
                "candidate_params": {
                    "l_freq": 4.0,
                    "h_freq": 40.0,
                    "apply_notch": False,
                    "apply_reference": True,
                    "epoch_config": {"tmin": 0.0, "tmax": 4.0, "baseline": None},
                },
            }
        )
    )

    with pytest.raises(ValueError, match="only test subjects"):
        held_out_evaluation.evaluate_held_out(subjects=(1, 2), selection_path=selection_path)

    with pytest.raises(ValueError, match="only test subjects"):
        held_out_evaluation.evaluate_held_out(subjects=(37, 38), selection_path=selection_path)

    with pytest.raises(ValueError, match="only test subjects"):
        held_out_evaluation.evaluate_held_out(
            subjects=(46, 47, 1), selection_path=selection_path
        )


def test_evaluate_held_out_requires_frozen_selection_file(tmp_path):
    from calm import held_out_evaluation

    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        held_out_evaluation.evaluate_held_out(subjects=(46, 47), selection_path=missing_path)


def test_evaluate_held_out_has_no_candidate_selection_parameter():
    """Evaluation cannot request a preprocessing sweep."""
    import inspect

    from calm import held_out_evaluation

    sig = inspect.signature(held_out_evaluation.evaluate_held_out)
    assert "candidates" not in sig.parameters
    assert "candidate" not in sig.parameters
