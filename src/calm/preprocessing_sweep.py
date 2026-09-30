"""Compare preprocessing settings, changing one setting at a time."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from calm.baseline_runner import run_baseline_evaluation
from calm.config import DATA_RAW_DIR, DEFAULT_SUBJECTS, LEFT_RIGHT_RUNS
from calm.signal.epochs import EpochConfig


@dataclass(frozen=True)
class PreprocessingCandidate:
    name: str
    l_freq: float = 4.0
    h_freq: float = 40.0
    apply_notch: bool = False
    apply_reference: bool = True
    epoch_config: EpochConfig = field(default_factory=lambda: EpochConfig(tmin=0.0, tmax=4.0))


DEFAULT_CANDIDATES: tuple[PreprocessingCandidate, ...] = (
    PreprocessingCandidate("default_car_4to40hz_no_notch_cue0s"),
    PreprocessingCandidate("no_common_average_reference", apply_reference=False),
    PreprocessingCandidate("wide_band_1to40hz", l_freq=1.0, h_freq=40.0),
    PreprocessingCandidate("with_60hz_notch", apply_notch=True),
    PreprocessingCandidate(
        "cue_delayed_1s", epoch_config=EpochConfig(tmin=1.0, tmax=5.0)
    ),
)


def run_preprocessing_sweep(
    subjects: tuple[int, ...] = DEFAULT_SUBJECTS,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
    candidates: tuple[PreprocessingCandidate, ...] = DEFAULT_CANDIDATES,
) -> pd.DataFrame:
    """Run the full baseline evaluation once per candidate; return one
    concatenated aggregate-results table tagged by candidate name."""
    rows = []
    for candidate in candidates:
        _, aggregate = run_baseline_evaluation(
            subjects=subjects,
            runs=runs,
            data_dir=data_dir,
            l_freq=candidate.l_freq,
            h_freq=candidate.h_freq,
            apply_notch=candidate.apply_notch,
            apply_reference=candidate.apply_reference,
            epoch_config=candidate.epoch_config,
        )
        aggregate = aggregate.copy()
        aggregate.insert(0, "preprocessing_candidate", candidate.name)
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True)
