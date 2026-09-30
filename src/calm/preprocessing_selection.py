"""Select preprocessing on validation subjects and save the choice to JSON."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from calm.config import DATA_RAW_DIR, LEFT_RIGHT_RUNS, RESULTS_DIR, VALIDATION_SUBJECTS
from calm.preprocessing_sweep import (
    DEFAULT_CANDIDATES,
    PreprocessingCandidate,
    run_preprocessing_sweep,
)

logger = logging.getLogger(__name__)

SELECTION_MODELS = ("bandpower_logreg", "csp_lda")  # majority is always 0.5, uninformative
DEFAULT_SELECTION_PATH = RESULTS_DIR / "preprocessing_selection.json"


def select_preprocessing_candidate(
    subjects: tuple[int, ...] = VALIDATION_SUBJECTS,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
    candidates: tuple[PreprocessingCandidate, ...] = DEFAULT_CANDIDATES,
    output_path: Path = DEFAULT_SELECTION_PATH,
) -> dict:
    subjects_set = set(subjects)
    if not subjects_set:
        raise ValueError("subjects must be non-empty")
    if not subjects_set.issubset(VALIDATION_SUBJECTS):
        raise ValueError(
            f"Preprocessing selection must use only validation subjects "
            f"{VALIDATION_SUBJECTS}; got {sorted(subjects_set)}. Selecting "
            "preprocessing on development or held-out-test subjects would "
            "leak those subjects into the choice of preprocessing."
        )

    sweep = run_preprocessing_sweep(
        subjects=subjects, runs=runs, data_dir=data_dir, candidates=candidates
    )
    real = sweep[(~sweep["shuffled_control"]) & (sweep["model"].isin(SELECTION_MODELS))]
    scores = (
        real.groupby("preprocessing_candidate")["mean_balanced_accuracy"]
        .mean()
        .sort_values(ascending=False)
    )
    chosen_name = scores.index[0]
    chosen_candidate = next(c for c in candidates if c.name == chosen_name)

    result = {
        "chosen_candidate": chosen_name,
        "selection_subjects": sorted(subjects_set),
        "selection_metric": (
            f"mean balanced_accuracy averaged over {SELECTION_MODELS} (real labels only)"
        ),
        "candidate_scores": scores.to_dict(),
        "candidate_params": dataclasses.asdict(chosen_candidate),
        "full_sweep": sweep.to_dict(orient="records"),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Selected preprocessing candidate '%s' -> %s", chosen_name, output_path)
    return result
