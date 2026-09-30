"""Record file checksums, recording metadata, and basic signal quality."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from calm.config import DATA_RAW_DIR, DEFAULT_SUBJECTS, LEFT_RIGHT_RUNS, RESULTS_DIR
from calm.data.download import download_eegbci_subjects
from calm.data.events import original_annotation_counts

logger = logging.getLogger(__name__)

FLAT_CHANNEL_VARIANCE_THRESHOLD = 1e-20  # volts^2; near-zero-variance channel


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_from_edf_path(path: Path) -> int:
    """Parse the run number out of a PhysioNet filename like S001R04.edf."""
    stem = path.stem  # e.g. "S001R04"
    run_token = stem.split("R")[-1]
    return int(run_token)


def audit_raw(
    raw: mne.io.BaseRaw,
    subject: int,
    run: int,
    file_path: Path,
    reference_channels: list[str] | None = None,
) -> dict:
    """Compute one audit-manifest row for an already-loaded Raw object."""
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    ch_types = list(raw.get_channel_types())
    n_channels = len(ch_names)
    duration_sec = float(raw.n_times / sfreq)

    annotation_counts = original_annotation_counts(raw)

    missing_channels: list[str] = []
    if reference_channels is not None:
        missing_channels = sorted(set(reference_channels) - set(ch_names))

    data = raw.get_data()  # (n_channels, n_times), volts
    per_channel_variance = data.var(axis=1)
    flat_channels = [
        ch_names[i]
        for i, v in enumerate(per_channel_variance)
        if v < FLAT_CHANNEL_VARIANCE_THRESHOLD
    ]
    has_nan = bool(np.isnan(data).any())
    has_inf = bool(np.isinf(data).any())
    max_abs_amplitude = float(np.max(np.abs(data))) if data.size else float("nan")

    return {
        "subject": subject,
        "run": run,
        "file_path": str(file_path),
        "file_sha256": _sha256_of_file(file_path),
        "sfreq_hz": sfreq,
        "n_channels": n_channels,
        "channel_names": ",".join(ch_names),
        "channel_types": ",".join(sorted(set(ch_types))),
        "duration_sec": duration_sec,
        "n_annotations": int(sum(annotation_counts.values())),
        "annotation_counts": dict(annotation_counts),
        "missing_channels": missing_channels,
        "n_missing_channels": len(missing_channels),
        "flat_channels": flat_channels,
        "n_flat_channels": len(flat_channels),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "max_abs_amplitude_v": max_abs_amplitude,
        "min_channel_variance_v2": (
            float(per_channel_variance.min()) if data.size else float("nan")
        ),
        "max_channel_variance_v2": (
            float(per_channel_variance.max()) if data.size else float("nan")
        ),
        "mean_channel_variance_v2": (
            float(per_channel_variance.mean()) if data.size else float("nan")
        ),
    }


def build_manifest(
    subjects: tuple[int, ...] = DEFAULT_SUBJECTS,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
) -> pd.DataFrame:
    """Download (if needed) and audit every subject-run, returning a DataFrame."""
    subject_paths = download_eegbci_subjects(subjects=subjects, runs=runs, data_dir=data_dir)

    rows: list[dict] = []
    reference_channels: list[str] | None = None
    for subject in subjects:
        for path in subject_paths[subject]:
            run = _run_from_edf_path(path)
            if run not in runs:
                continue
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            if reference_channels is None:
                reference_channels = list(raw.ch_names)
            row = audit_raw(raw, subject, run, path, reference_channels=reference_channels)
            rows.append(row)
            logger.info(
                "Audited subject=%d run=%d -> %d events", subject, run, row["n_annotations"]
            )

    manifest = pd.DataFrame(rows).sort_values(["subject", "run"]).reset_index(drop=True)
    return manifest


def save_manifest(manifest: pd.DataFrame, path: Path = RESULTS_DIR / "audit_manifest.csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path
