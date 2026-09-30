from __future__ import annotations

from pathlib import Path

import pytest

from calm.audit.manifest import _run_from_edf_path, _sha256_of_file, audit_raw


@pytest.mark.parametrize(
    ("filename", "expected_run"),
    [("S001R04.edf", 4), ("S005R12.edf", 12), ("S109R08.edf", 8)],
)
def test_run_from_edf_path(filename, expected_run):
    assert _run_from_edf_path(Path(filename)) == expected_run


def test_sha256_is_deterministic(tmp_path):
    f = tmp_path / "dummy.edf"
    f.write_bytes(b"synthetic-edf-bytes")
    assert _sha256_of_file(f) == _sha256_of_file(f)
    assert len(_sha256_of_file(f)) == 64


def test_audit_raw_basic_fields(tmp_path, synthetic_raw_run4):
    dummy_file = tmp_path / "S001R04.edf"
    dummy_file.write_bytes(b"placeholder")

    row = audit_raw(synthetic_raw_run4, subject=1, run=4, file_path=dummy_file)

    assert row["subject"] == 1
    assert row["run"] == 4
    assert row["sfreq_hz"] == 160.0
    assert row["n_channels"] == 8
    assert row["duration_sec"] == pytest.approx(20.0, abs=0.01)
    assert row["n_annotations"] == 6
    assert row["annotation_counts"] == {"T0": 3, "T1": 2, "T2": 1}
    assert row["n_flat_channels"] == 0
    assert row["has_nan"] is False
    assert row["has_inf"] is False
    assert row["max_abs_amplitude_v"] > 0


def test_audit_raw_detects_flat_channel(tmp_path, synthetic_raw_with_flat_channel):
    dummy_file = tmp_path / "S001R06.edf"
    dummy_file.write_bytes(b"placeholder")

    row = audit_raw(synthetic_raw_with_flat_channel, subject=1, run=6, file_path=dummy_file)

    assert row["n_flat_channels"] == 1
    assert synthetic_raw_with_flat_channel.ch_names[0] in row["flat_channels"]


def test_audit_raw_detects_missing_channels(tmp_path, synthetic_raw_run4):
    dummy_file = tmp_path / "S001R04.edf"
    dummy_file.write_bytes(b"placeholder")

    reference_channels = list(synthetic_raw_run4.ch_names) + ["ExtraChannel"]
    row = audit_raw(
        synthetic_raw_run4,
        subject=1,
        run=4,
        file_path=dummy_file,
        reference_channels=reference_channels,
    )

    assert row["missing_channels"] == ["ExtraChannel"]
    assert row["n_missing_channels"] == 1
