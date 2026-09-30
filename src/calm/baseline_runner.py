"""Load and preprocess each run, then evaluate classical cross-run baselines."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import mne
import pandas as pd

from calm.config import DATA_RAW_DIR, DEFAULT_SUBJECTS, LEFT_RIGHT_RUNS
from calm.data.download import download_eegbci_subjects
from calm.eval.cross_run import ModelSpec, aggregate_results, within_subject_cross_run
from calm.features.bandpower import bandpower_features
from calm.models.baselines import (
    make_bandpower_logreg_pipeline,
    make_csp_lda_pipeline,
    make_majority_baseline,
)
from calm.signal.epochs import EpochConfig, epochs_to_arrays, left_right_epochs_from_raw
from calm.signal.filtering import bandpass_filter_offline, common_average_reference, notch_filter

logger = logging.getLogger(__name__)


def build_subject_run_data(
    subject: int,
    runs: tuple[int, ...],
    data_dir: Path,
    l_freq: float = 4.0,
    h_freq: float = 40.0,
    apply_notch: bool = False,
    apply_reference: bool = True,
    epoch_config: EpochConfig | None = None,
) -> dict[int, tuple]:
    epoch_config = epoch_config or EpochConfig()
    subject_paths = download_eegbci_subjects(subjects=(subject,), runs=runs, data_dir=data_dir)
    run_data = {}
    for path in sorted(subject_paths[subject]):
        run = int(path.stem.split("R")[-1])
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        if apply_reference:
            raw = common_average_reference(raw)
        raw = bandpass_filter_offline(raw, l_freq, h_freq)
        if apply_notch:
            raw = notch_filter(raw)
        epochs = left_right_epochs_from_raw(raw, run, config=epoch_config)
        X, y = epochs_to_arrays(epochs)
        run_data[run] = (X, y)
        logger.info("subject=%d run=%d epochs=%d", subject, run, len(y))
    return run_data


def run_baseline_evaluation(
    subjects: tuple[int, ...] = DEFAULT_SUBJECTS,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
    l_freq: float = 4.0,
    h_freq: float = 40.0,
    apply_notch: bool = False,
    apply_reference: bool = True,
    epoch_config: EpochConfig | None = None,
    csp_components: int = 6,
    extra_specs: list[ModelSpec] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bandpower_fn = functools.partial(bandpower_features, sfreq=160.0)
    specs = [
        ModelSpec("majority", make_majority_baseline, feature_fn=None),
        ModelSpec(
            "bandpower_logreg", make_bandpower_logreg_pipeline, feature_fn=bandpower_fn
        ),
        ModelSpec(
            "csp_lda", lambda: make_csp_lda_pipeline(n_components=csp_components), feature_fn=None
        ),
        *(extra_specs or []),
    ]

    all_results: list[dict] = []
    for subject in subjects:
        run_data = build_subject_run_data(
            subject,
            runs,
            data_dir,
            l_freq=l_freq,
            h_freq=h_freq,
            apply_notch=apply_notch,
            apply_reference=apply_reference,
            epoch_config=epoch_config,
        )
        for spec in specs:
            all_results.extend(within_subject_cross_run(subject, run_data, spec))
            all_results.extend(
                within_subject_cross_run(subject, run_data, spec, shuffled_control=True)
            )

    per_fold = pd.DataFrame(all_results)
    aggregate = aggregate_results(all_results)
    return per_fold, aggregate
