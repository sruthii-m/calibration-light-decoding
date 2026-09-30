"""Compare clean and zero-filled channel-dropout performance on test runs.

Only random channel dropout is evaluated here."""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from calm.baseline_runner import build_subject_run_data
from calm.config import DATA_RAW_DIR, LEFT_RIGHT_RUNS, TEST_SUBJECTS
from calm.eval.cross_run import ModelSpec
from calm.features.bandpower import bandpower_features
from calm.mae.evaluate import pretrained_encoder_spec
from calm.models.baselines import make_bandpower_logreg_pipeline, make_csp_lda_pipeline
from calm.preprocessing_selection import DEFAULT_SELECTION_PATH
from calm.robustness.corruptions import random_channel_dropout
from calm.signal.epochs import EpochConfig

logger = logging.getLogger(__name__)


def _classical_specs(csp_components: int = 6) -> list[ModelSpec]:
    bandpower_fn = functools.partial(bandpower_features, sfreq=160.0)
    return [
        ModelSpec("bandpower_logreg", make_bandpower_logreg_pipeline, feature_fn=bandpower_fn),
        ModelSpec(
            "csp_lda", lambda: make_csp_lda_pipeline(n_components=csp_components), feature_fn=None
        ),
    ]


def evaluate_channel_dropout_robustness(
    subjects: tuple[int, ...] = TEST_SUBJECTS,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    encoder_checkpoints: dict[str, Path] | None = None,
    dropout_fraction: float = 0.25,
    seed: int = 0,
) -> pd.DataFrame:
    if not set(subjects).issubset(TEST_SUBJECTS):
        raise ValueError(
            f"Robustness evaluation must use only test subjects {TEST_SUBJECTS}; got {subjects}"
        )

    selection = json.loads(Path(selection_path).read_text())
    params = selection["candidate_params"]
    epoch_cfg = params["epoch_config"]

    specs = _classical_specs()
    for name, checkpoint_path in (encoder_checkpoints or {}).items():
        specs.append(pretrained_encoder_spec(checkpoint_path, name))

    rows = []
    for subject in subjects:
        run_data = build_subject_run_data(
            subject,
            runs,
            data_dir,
            l_freq=params["l_freq"],
            h_freq=params["h_freq"],
            apply_notch=params["apply_notch"],
            apply_reference=params["apply_reference"],
            epoch_config=EpochConfig(tmin=epoch_cfg["tmin"], tmax=epoch_cfg["tmax"]),
        )
        for held_out_run in sorted(run_data):
            train_runs = [r for r in run_data if r != held_out_run]
            X_train = np.concatenate([run_data[r][0] for r in train_runs], axis=0)
            y_train = np.concatenate([run_data[r][1] for r in train_runs], axis=0)
            X_test_clean, y_test = run_data[held_out_run]
            X_test_corrupted, corruption_log = random_channel_dropout(
                X_test_clean, fraction=dropout_fraction, seed=seed + subject * 100 + held_out_run
            )

            for spec in specs:
                X_train_feat = spec.feature_fn(X_train) if spec.feature_fn else X_train
                X_clean_feat = spec.feature_fn(X_test_clean) if spec.feature_fn else X_test_clean
                X_corrupt_feat = (
                    spec.feature_fn(X_test_corrupted) if spec.feature_fn else X_test_corrupted
                )

                model = spec.model_factory()
                model.fit(X_train_feat, y_train)
                clean_acc = balanced_accuracy_score(y_test, model.predict(X_clean_feat))
                corrupt_acc = balanced_accuracy_score(y_test, model.predict(X_corrupt_feat))

                rows.append(
                    {
                        "subject": subject,
                        "held_out_run": held_out_run,
                        "model": spec.name,
                        "dropped_channels": corruption_log.affected_channels,
                        "n_dropped_channels": len(corruption_log.affected_channels),
                        "clean_balanced_accuracy": clean_acc,
                        "corrupted_balanced_accuracy": corrupt_acc,
                        "degradation": clean_acc - corrupt_acc,
                    }
                )
                logger.info(
                    "subject=%d run=%d model=%s clean=%.3f corrupted=%.3f",
                    subject, held_out_run, spec.name, clean_acc, corrupt_acc,
                )

    return pd.DataFrame(rows)
