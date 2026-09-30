"""Paths, subject splits, and supported motor-imagery runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
RESULTS_DIR = ARTIFACTS_DIR / "results"

# Left-vs-right motor imagery runs. Runs 6/10/14 (both-hands vs both-feet)
# are intentionally excluded.
LEFT_RIGHT_RUNS = (4, 8, 12)

# Small default for downloads and quick baseline checks.
DEFAULT_SUBJECTS = (1, 2, 3, 4, 5)

# Expanded from the original 20-subject pilot; keep all three groups disjoint.
DEV_SUBJECTS = tuple(range(1, 37))
VALIDATION_SUBJECTS = tuple(range(37, 46))
TEST_SUBJECTS = tuple(range(46, 61))

# Limit the size of a single download request.
MAX_PILOT_SUBJECTS = 60

EXPECTED_SFREQ_HZ = 160.0
EXPECTED_N_CHANNELS = 64


def assert_disjoint_subject_splits() -> None:
    dev, val, test = set(DEV_SUBJECTS), set(VALIDATION_SUBJECTS), set(TEST_SUBJECTS)
    overlaps = {
        "dev∩validation": dev & val,
        "dev∩test": dev & test,
        "validation∩test": val & test,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"Subject splits are not disjoint: {bad}")


assert_disjoint_subject_splits()


@dataclass(frozen=True)
class AuditConfig:
    subjects: tuple[int, ...] = DEFAULT_SUBJECTS
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS
    raw_data_dir: Path = field(default=DATA_RAW_DIR)
    manifest_path: Path = field(default=RESULTS_DIR / "audit_manifest.csv")

    def __post_init__(self) -> None:
        validate_subjects(self.subjects)
        validate_runs(self.runs)


def validate_subjects(subjects: tuple[int, ...]) -> None:
    if not subjects:
        raise ValueError("subjects must be non-empty")
    if any(s < 1 or s > 109 for s in subjects):
        raise ValueError("PhysioNet EEGBCI subjects are numbered 1-109")
    if len(subjects) > MAX_PILOT_SUBJECTS:
        raise ValueError(
            f"Refusing to silently process {len(subjects)} subjects "
            f"(> MAX_PILOT_SUBJECTS={MAX_PILOT_SUBJECTS}). "
            "Raise MAX_PILOT_SUBJECTS deliberately if this is intended."
        )


def validate_runs(runs: tuple[int, ...]) -> None:
    if not runs:
        raise ValueError("runs must be non-empty")
    invalid = set(runs) - set(LEFT_RIGHT_RUNS)
    if invalid:
        raise ValueError(
            f"Runs {sorted(invalid)} are not left/right motor-imagery runs. "
            f"Only {LEFT_RIGHT_RUNS} map T1/T2 to left/right hand imagery; "
            "runs 6, 10, 14 are both-hands vs both-feet."
        )
