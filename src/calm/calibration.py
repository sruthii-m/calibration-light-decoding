"""Exploratory calibration curves using fixed representations and paired trial samples.

No encoder training or test-dependent hyperparameter selection. Seeds repeat trial
selection, not pretraining. Subjects, not rotations or seeds, are inference units.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from calm.baseline_runner import build_subject_run_data
from calm.config import DEV_SUBJECTS, PROJECT_ROOT, TEST_SUBJECTS, VALIDATION_SUBJECTS
from calm.eval.cross_run import _score_of
from calm.eval.metrics import compute_metrics
from calm.features.bandpower import bandpower_features
from calm.mae.probe import build_encoder_from_checkpoint, load_checkpoint
from calm.models.baselines import (
    make_bandpower_logreg_pipeline,
    make_csp_lda_pipeline,
    make_majority_baseline,
)
from calm.representation.stft_frontend import STFTConfig, epochs_to_log_power
from calm.representation.tokens import pool_to_band_time_tokens
from calm.signal.epochs import EpochConfig

LOG = logging.getLogger(__name__)
ENCODERS = ("random_encoder", "pretrained_encoder", "channel_encoder")
MODELS = ("majority", "bandpower", "csp_lda", "stft_flat", "tokens_flat", *ENCODERS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_indices(
    y: np.ndarray, budget: int | None, seed: int, subject: int, held_out_run: int
) -> np.ndarray:
    """Nested, class-balanced subsets; full budget retains every available trial."""
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibration requires both binary classes")
    if budget is None:
        return np.arange(len(y))
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, subject, held_out_run]))
    selected = []
    for label in (0, 1):
        candidates = np.flatnonzero(y == label)
        if len(candidates) < budget:
            raise ValueError(f"Only {len(candidates)} trials for class {label}; requested {budget}")
        selected.extend(rng.permutation(candidates)[:budget].tolist())
    return np.sort(selected)


def fit_probe(
    name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, c: float
) -> tuple[object, list, list]:
    if name == "majority":
        model = make_majority_baseline()
    elif name == "csp_lda":
        model = make_csp_lda_pipeline(n_components=6)
    elif name.endswith("_unscaled"):
        model = LogisticRegression(C=c, max_iter=2000)
    else:
        model = make_bandpower_logreg_pipeline(C=c, max_iter=2000)
    # A convergence failure invalidates the run rather than silently emitting results.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = _score_of(model, X_test)
    return model, pred.tolist(), score.tolist()


def subject_results(rows: pd.DataFrame) -> pd.DataFrame:
    # Average repeated selections within each rotation first, then rotations.
    keys = ["subject", "model", "budget"]
    metrics = ["balanced_accuracy", "roc_auc", "cohen_kappa"]
    folds = rows.groupby([*keys, "held_out_run"])[metrics].mean().reset_index()
    return folds.groupby(keys)[metrics].mean().reset_index()


def interval(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(subjects: pd.DataFrame, n_boot: int, seed: int):
    aggregate = []
    for (model, budget), group in subjects.groupby(["model", "budget"]):
        lo, hi = interval(group.balanced_accuracy.to_numpy(), n_boot, seed)
        aggregate.append(
            dict(
                model=model,
                budget=budget,
                n_subjects=len(group),
                balanced_accuracy=group.balanced_accuracy.mean(),
                ci_low=lo,
                ci_high=hi,
                roc_auc=group.roc_auc.mean(),
                cohen_kappa=group.cohen_kappa.mean(),
            )
        )
    paired = []
    for budget in ("5", "10", "all"):
        wide = subjects[subjects.budget == budget].pivot(
            index="subject", columns="model", values="balanced_accuracy"
        )
        for model in MODELS:
            for reference in ("random_encoder", "tokens_flat", "csp_lda"):
                if model == reference:
                    continue
                for subject, delta in (wide[model] - wide[reference]).items():
                    paired.append(
                        dict(
                            subject=subject,
                            model=model,
                            budget=budget,
                            reference=reference,
                            reference_budget=budget,
                            delta=delta,
                        )
                    )
        full = subjects[(subjects.budget == "all") & (subjects.model == "csp_lda")]
        full = full.set_index("subject").balanced_accuracy
        if budget != "all":
            for model in ENCODERS:
                for subject, delta in (wide[model] - full).items():
                    paired.append(
                        dict(
                            subject=subject,
                            model=model,
                            budget=budget,
                            reference="csp_lda",
                            reference_budget="all",
                            delta=delta,
                        )
                    )
    # Direct scaling diagnostic, computed on identical features and full-data folds.
    wide = subjects[subjects.budget == "all"].pivot(
        index="subject", columns="model", values="balanced_accuracy"
    )
    for model in ENCODERS:
        for subject, delta in (wide[model] - wide[model + "_unscaled"]).items():
            paired.append(
                dict(
                    subject=subject,
                    model=model,
                    budget="all",
                    reference=model + "_unscaled",
                    reference_budget="all",
                    delta=delta,
                )
            )
    paired = pd.DataFrame(paired)
    comparisons = []
    for keys, group in paired.groupby(["model", "budget", "reference", "reference_budget"]):
        lo, hi = interval(group.delta.to_numpy(), n_boot, seed)
        comparisons.append(
            dict(
                zip(["model", "budget", "reference", "reference_budget"], keys, strict=True),
                delta=group.delta.mean(),
                ci_low=lo,
                ci_high=hi,
                n_subjects=len(group),
                wins=int((group.delta > 1e-12).sum()),
                ties=int(np.isclose(group.delta, 0).sum()),
            )
        )
    return pd.DataFrame(aggregate), paired, pd.DataFrame(comparisons)


def validate_artifacts(config: dict, selection: dict, checkpoints: list[dict]) -> None:
    subjects = config["subjects"]
    if (
        not subjects
        or len(set(subjects)) != len(subjects)
        or not set(subjects) <= set(TEST_SUBJECTS)
    ):
        raise ValueError("Require unique evaluation subjects from TEST_SUBJECTS")
    if set(config["runs"]) != {4, 8, 12} or len(config["runs"]) != 3:
        raise ValueError("Require exactly runs 4, 8, 12")
    if config["budgets"] != [5, 10, "all"]:
        raise ValueError("This protocol requires budgets 5, 10, all")
    if not config["sampling_seeds"] or len(set(config["sampling_seeds"])) != len(
        config["sampling_seeds"]
    ):
        raise ValueError("Require nonempty unique sampling seeds")
    if not selection["selection_subjects"] or not set(selection["selection_subjects"]) <= set(
        VALIDATION_SUBJECTS
    ):
        raise ValueError("Selection provenance must use validation subjects")
    params = {k: v for k, v in selection["candidate_params"].items() if k != "name"}
    for checkpoint in checkpoints:
        trained = set(checkpoint["pretraining_subjects"])
        if not trained or not trained <= set(DEV_SUBJECTS) or trained & set(subjects):
            raise ValueError("Checkpoint pretraining subject leakage or invalid provenance")
        if checkpoint["preprocessing"] != params:
            raise ValueError("Checkpoint and evaluation preprocessing differ")
    for key in ("geometry", "stft_config", "mae_config"):
        if checkpoints[0][key] != checkpoints[1][key]:
            raise ValueError(f"Checkpoints have incompatible {key}")
    for key in ("normalizer_mean", "normalizer_std"):
        np.testing.assert_array_equal(checkpoints[0][key], checkpoints[1][key])
    if [c["mask_strategy"] for c in checkpoints] != ["random", "whole_channel"]:
        raise ValueError("Expected random and whole-channel checkpoints in that order")


def write_report(out: Path, aggregate: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    lines = [
        "# CALM exploratory calibration experiment",
        "",
        "Fixed existing checkpoints; no encoder retraining or hyperparameter search. "
        "Subjects were previously inspected. Each classifier is calibrated on the same "
        "subject's two other runs. Seeds repeat trial selection, not encoder training.",
        "",
        "All logistic probes use calibration-only StandardScaler and C=1, except the "
        "explicit unscaled diagnostic. Full STFT is flattened log-power up to 40 Hz; "
        "tokens_flat uses the exact mu/beta × time pooling supplied to the encoder.",
        "",
        "| Model | 5/class | 10/class | All (30 trials) |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        entries = aggregate[aggregate.model == model].set_index("budget")
        cells = [
            f"{entries.loc[b, 'balanced_accuracy'] * 100:.2f} "
            f"[{entries.loc[b, 'ci_low'] * 100:.2f}, "
            f"{entries.loc[b, 'ci_high'] * 100:.2f}]"
            for b in ("5", "10", "all")
        ]
        lines.append("| " + " | ".join([model, *cells]) + " |")
    lines += [
        "",
        "Balanced accuracy (%), subject-bootstrap 95% intervals. "
        "Average sampling seeds within rotations, then rotations within subjects. "
        "Intervals do not treat seeds/folds as independent or capture pretraining variability.",
        "",
        "## Paired encoder comparisons",
        "",
        "| Model | Budget | Reference | Reference budget | Difference (pp), 95% CI |",
        "|---|---|---|---|---:|",
    ]
    for row in comparisons.itertuples():
        if row.model not in ENCODERS:
            continue
        lines.append(
            f"| {row.model} | {row.budget} | {row.reference} | "
            f"{row.reference_budget} | {row.delta * 100:+.2f} "
            f"[{row.ci_low * 100:+.2f}, {row.ci_high * 100:+.2f}] |"
        )
    lines += [
        "",
        "These are exploratory, unadjusted intervals for specified comparisons, "
        "not equivalence tests or proof of motor-specific/clinical/online decoding. "
        "The full calibration pool may be slightly class-imbalanced. "
        "Five-trial selections are nested within ten-trial selections. "
        "No best-seed selection is used.",
        "",
        "See protocol.json for source/checkpoint/data hashes and exact configuration; "
        "selections.jsonl for shared calibration IDs; per_fold.csv and predictions.jsonl "
        "for all fits; paired_per_subject.csv for subject-level differences.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        rows = aggregate[aggregate.model == model].set_index("budget").loc[["5", "10", "all"]]
        ax.plot([10, 20, 30], rows.balanced_accuracy * 100, marker="o", label=model)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Total labeled calibration trials",
        ylabel="Mean balanced accuracy (%)",
        title="CALM: exploratory cross-run calibration curves",
        xticks=[10, 20, 30],
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "calibration_curve.png", dpi=180)
    plt.close(fig)


def run_calibration(config_path: Path, output_dir: Path | None = None) -> Path:
    config = yaml.safe_load(config_path.read_text())
    root = PROJECT_ROOT
    out = output_dir or root / config["output_dir"]
    selection_path = root / config["selection"]
    selection = json.loads(selection_path.read_text())
    paths = [root / config[k] for k in ("random_mask_checkpoint", "channel_mask_checkpoint")]
    checkpoints = [load_checkpoint(p) for p in paths]
    validate_artifacts(config, selection, checkpoints)
    manifest_path = root / config["manifest"]
    manifest = pd.read_csv(manifest_path)
    subset = manifest[manifest.subject.isin(config["subjects"]) & manifest.run.isin(config["runs"])]
    if len(subset) != len(config["subjects"]) * 3 or subset.duplicated(["subject", "run"]).any():
        raise ValueError("Manifest does not cover every evaluation recording exactly once")
    for row in subset.itertuples():
        if (
            row.sfreq_hz != 160
            or row.channel_names.split(",") != checkpoints[0]["geometry"]["ch_names"]
        ):
            raise ValueError("Sampling frequency or channel order differs from checkpoint")
        if sha256(Path(row.file_path)) != row.file_sha256:
            raise ValueError(f"Raw file checksum mismatch: {row.file_path}")
    out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    metadata = dict(
        config=config,
        status="running",
        exploratory=True,
        started_utc=datetime.now(UTC).isoformat(),
        git_commit=revision.stdout.strip() if revision.returncode == 0 else None,
        source_sha256={
            str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))
        },
        config_sha256=sha256(config_path),
        selection=selection,
        selection_sha256=sha256(selection_path),
        manifest_sha256=sha256(manifest_path),
        checkpoint_sha256={str(p.relative_to(root)): sha256(p) for p in paths},
        dependencies={
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "mne", "torch")
        },
        splits=dict(
            dev=list(DEV_SUBJECTS), validation=list(VALIDATION_SUBJECTS), test=list(TEST_SUBJECTS)
        ),
        limitations=[
            "Previously inspected evaluation subjects",
            "One existing checkpoint per masking strategy",
            "Fixed C=1; no regularization search",
            "No new shuffled-label experiment; prior controls retained",
        ],
    )
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2))
    (out / "config.yaml").write_text(config_path.read_text())
    subset.to_csv(out / "dataset_manifest.csv", index=False)
    torch.set_num_threads(1)
    mne.set_log_level("ERROR")
    encoded = {}
    for name, checkpoint, pretrained in [
        (ENCODERS[0], checkpoints[0], False),
        (ENCODERS[1], checkpoints[0], True),
        (ENCODERS[2], checkpoints[1], True),
    ]:
        encoded[name] = build_encoder_from_checkpoint(
            checkpoint, pretrained, seed=config["encoder_seed"]
        )
    stft_config = STFTConfig(**checkpoints[0]["stft_config"])
    geometry = encoded[ENCODERS[0]][1]
    params = {k: v for k, v in selection["candidate_params"].items() if k != "name"}
    params["epoch_config"] = EpochConfig(**params["epoch_config"])
    all_rows, feature_diagnostics = [], []
    with (
        (out / "selections.jsonl").open("w") as selections,
        (out / "predictions.jsonl").open("w") as predictions,
        threadpool_limits(limits=1),
    ):
        for subject in config["subjects"]:
            run_data = build_subject_run_data(
                subject, tuple(config["runs"]), root / config["data_dir"], **params
            )
            features = {}
            for run, (X, _y) in run_data.items():
                power, freqs, _ = epochs_to_log_power(X, 160.0, stft_config)
                tokens = pool_to_band_time_tokens(
                    power, freqs, bands=geometry.bands, n_time_bins=geometry.n_time_bins
                )
                values = dict(
                    majority=X,
                    csp_lda=X,
                    bandpower=bandpower_features(X, 160.0),
                    stft_flat=power.reshape(len(X), -1),
                    tokens_flat=tokens.reshape(len(X), -1),
                )
                for name, (model, _geometry, normalizer) in encoded.items():
                    normalized = torch.from_numpy(normalizer.transform(tokens).astype(np.float32))
                    values[name] = model.embed(normalized).numpy()
                    values[name + "_unscaled"] = values[name]
                features[run] = values
                for name in MODELS[3:]:
                    feature_diagnostics.append(
                        dict(
                            subject=subject,
                            run=run,
                            model=name,
                            n_features=values[name].shape[1],
                            mean_feature_std=float(values[name].std(0).mean()),
                        )
                    )
            for held_out in sorted(run_data):
                train_runs = sorted(set(run_data) - {held_out})
                y_train = np.concatenate([run_data[r][1] for r in train_runs])
                y_test = run_data[held_out][1]
                train_ids = [
                    f"S{subject:03d}R{r:02d}:epoch{i}"
                    for r in train_runs
                    for i in range(len(run_data[r][1]))
                ]
                test_ids = [f"S{subject:03d}R{held_out:02d}:epoch{i}" for i in range(len(y_test))]
                train_features = {
                    name: np.concatenate([features[r][name] for r in train_runs])
                    for name in features[held_out]
                }
                for budget in config["budgets"]:
                    seeds = [-1] if budget == "all" else config["sampling_seeds"]
                    for seed in seeds:
                        idx = calibration_indices(
                            y_train, None if budget == "all" else budget, seed, subject, held_out
                        )
                        sample_id = f"{subject}-{held_out}-{budget}-{seed}"
                        selections.write(
                            json.dumps(
                                dict(
                                    sample_id=sample_id,
                                    subject=subject,
                                    held_out_run=held_out,
                                    train_runs=train_runs,
                                    budget=str(budget),
                                    seed=seed,
                                    calibration_ids=[train_ids[i] for i in idx],
                                    calibration_labels=y_train[idx].tolist(),
                                    test_ids=test_ids,
                                )
                            )
                            + "\n"
                        )
                        names = (
                            (*MODELS, *(n + "_unscaled" for n in ENCODERS))
                            if (budget == "all")
                            else MODELS
                        )
                        for name in names:
                            _model, pred, score = fit_probe(
                                name,
                                train_features[name][idx],
                                y_train[idx],
                                features[held_out][name],
                                config["probe_C"],
                            )
                            result = compute_metrics(y_test, np.asarray(pred), np.asarray(score))
                            result.update(
                                subject=subject,
                                held_out_run=held_out,
                                model=name,
                                budget=str(budget),
                                sampling_seed=seed,
                                sample_id=sample_id,
                                n_calibration=len(idx),
                                n_class_0=int((y_train[idx] == 0).sum()),
                                n_class_1=int((y_train[idx] == 1).sum()),
                            )
                            all_rows.append(result)
                            predictions.write(
                                json.dumps(
                                    dict(
                                        sample_id=sample_id,
                                        model=name,
                                        y_true=y_test.tolist(),
                                        y_pred=pred,
                                        score=score,
                                    )
                                )
                                + "\n"
                            )
            pd.DataFrame(all_rows).to_csv(out / "per_fold.csv", index=False)
            selections.flush()
            predictions.flush()
            LOG.info("Completed subject %s (%s fits total)", subject, len(all_rows))
    subjects = subject_results(pd.DataFrame(all_rows))
    aggregate, paired, comparisons = summarize(
        subjects, config["bootstrap_samples"], config["bootstrap_seed"]
    )
    for name, frame in [
        ("per_subject", subjects),
        ("aggregate", aggregate),
        ("paired_per_subject", paired),
        ("paired_comparisons", comparisons),
        ("feature_diagnostics", pd.DataFrame(feature_diagnostics)),
    ]:
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(aggregate.to_dict(orient="records"), indent=2))
    write_report(out, aggregate, comparisons)
    metadata.update(
        status="complete", completed_utc=datetime.now(UTC).isoformat(), n_fits=len(all_rows)
    )
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2))
    return out
