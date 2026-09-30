"""Download requested PhysioNet recordings, reusing cached files."""

from __future__ import annotations

import logging
from pathlib import Path

from mne.datasets import eegbci

from calm.config import (
    DATA_RAW_DIR,
    DEFAULT_SUBJECTS,
    LEFT_RIGHT_RUNS,
    validate_runs,
    validate_subjects,
)

logger = logging.getLogger(__name__)


def download_eegbci_subjects(
    subjects: tuple[int, ...] = DEFAULT_SUBJECTS,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
) -> dict[int, list[Path]]:
    """Download EDF files for the given subjects/runs into ``data_dir``.

    Returns a mapping of subject -> list of downloaded EDF file paths.
    """
    validate_subjects(subjects)
    validate_runs(runs)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading PhysioNet EEGBCI: subjects=%s runs=%s -> %s",
        list(subjects),
        list(runs),
        data_dir,
    )

    paths: dict[int, list[Path]] = {}
    for subject in subjects:
        subject_paths = eegbci.load_data(
            subjects=subject,
            runs=list(runs),
            path=str(data_dir),
            update_path=False,
            verbose=False,
        )
        paths[subject] = [Path(p) for p in subject_paths]
    return paths
