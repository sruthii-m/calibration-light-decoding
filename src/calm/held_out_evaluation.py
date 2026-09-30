"""Evaluate cross-run decoding on subjects excluded from pretraining.

Each classifier trains on two runs from the evaluation subject. Preprocessing
is loaded from the saved validation selection."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from calm.baseline_runner import run_baseline_evaluation
from calm.config import DATA_RAW_DIR, LEFT_RIGHT_RUNS, RESULTS_DIR, TEST_SUBJECTS
from calm.eval.cross_run import per_subject_results
from calm.mae.pretrain import DEFAULT_CHECKPOINT_PATH
from calm.preprocessing_selection import DEFAULT_SELECTION_PATH
from calm.signal.epochs import EpochConfig

logger = logging.getLogger(__name__)


def load_frozen_preprocessing(selection_path: Path = DEFAULT_SELECTION_PATH) -> dict:
    selection_path = Path(selection_path)
    if not selection_path.exists():
        raise FileNotFoundError(
            f"No frozen preprocessing selection at {selection_path}. Run "
            "`calm select-preprocessing` on validation subjects first."
        )
    return json.loads(selection_path.read_text())


def evaluate_held_out(
    subjects: tuple[int, ...] = TEST_SUBJECTS,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    mae_checkpoint_path: Path | None = DEFAULT_CHECKPOINT_PATH,
    output_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save and return fold, subject, and aggregate metrics.

    Include random and pretrained probes when a checkpoint is available."""
    subjects_set = set(subjects)
    if not subjects_set:
        raise ValueError("subjects must be non-empty")
    if not subjects_set.issubset(TEST_SUBJECTS):
        raise ValueError(
            f"Held-out evaluation must use only test subjects {TEST_SUBJECTS}; "
            f"got {sorted(subjects_set)}. Development/validation subjects must "
            "never appear in the final held-out evaluation."
        )

    selection = load_frozen_preprocessing(selection_path)
    params = selection["candidate_params"]
    epoch_cfg = params["epoch_config"]
    logger.info(
        "Held-out evaluation using frozen candidate '%s' (selected on subjects %s)",
        selection["chosen_candidate"],
        selection["selection_subjects"],
    )

    extra_specs = None
    if mae_checkpoint_path is not None and Path(mae_checkpoint_path).exists():
        from calm.mae.evaluate import encoder_model_specs

        extra_specs = encoder_model_specs(mae_checkpoint_path)
        logger.info("Including frozen-encoder probes from %s", mae_checkpoint_path)

    per_fold, aggregate = run_baseline_evaluation(
        subjects=subjects,
        runs=runs,
        data_dir=data_dir,
        l_freq=params["l_freq"],
        h_freq=params["h_freq"],
        apply_notch=params["apply_notch"],
        apply_reference=params["apply_reference"],
        epoch_config=EpochConfig(tmin=epoch_cfg["tmin"], tmax=epoch_cfg["tmax"]),
        extra_specs=extra_specs,
    )
    per_subject = per_subject_results(per_fold.to_dict(orient="records"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(output_dir / "held_out_per_fold.csv", index=False)
    per_subject.to_csv(output_dir / "held_out_per_subject.csv", index=False)
    aggregate.to_csv(output_dir / "held_out_aggregate.csv", index=False)

    return per_fold, per_subject, aggregate
