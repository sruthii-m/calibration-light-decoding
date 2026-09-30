"""Command-line entry points for data preparation and experiments."""

from __future__ import annotations

import argparse
import logging
import sys

from calm.config import (
    ARTIFACTS_DIR,
    DATA_RAW_DIR,
    DEFAULT_SUBJECTS,
    DEV_SUBJECTS,
    LEFT_RIGHT_RUNS,
    RESULTS_DIR,
    TEST_SUBJECTS,
    VALIDATION_SUBJECTS,
)


def _add_subjects_runs_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(DEFAULT_SUBJECTS),
        help="PhysioNet subject numbers (default: %(default)s)",
    )
    parser.add_argument(
        "--runs", type=int, nargs="+", default=list(LEFT_RIGHT_RUNS),
        help="Left/right motor-imagery runs (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_RAW_DIR),
        help="Directory to store/read raw EDF files (default: %(default)s)",
    )


def cmd_download(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.data.download import download_eegbci_subjects

    paths = download_eegbci_subjects(
        subjects=tuple(args.subjects), runs=tuple(args.runs), data_dir=Path(args.data_dir)
    )
    for subject, subject_paths in paths.items():
        print(f"subject {subject}: {len(subject_paths)} files")
        for p in subject_paths:
            print(f"  {p}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.audit.manifest import build_manifest, save_manifest

    manifest = build_manifest(
        subjects=tuple(args.subjects), runs=tuple(args.runs), data_dir=Path(args.data_dir)
    )
    out_path = save_manifest(manifest, path=Path(args.output))
    print(f"Wrote manifest with {len(manifest)} rows to {out_path}")
    print(manifest[["subject", "run", "sfreq_hz", "n_channels", "duration_sec", "n_annotations"]])
    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.viz.explore_subject import generate_subject_figures

    saved = generate_subject_figures(
        subject=args.subject,
        runs=tuple(args.runs),
        data_dir=Path(args.data_dir),
    )
    print(f"Saved {len(saved)} figures:")
    for p in saved:
        print(f"  {p}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.baseline_runner import run_baseline_evaluation

    per_fold, aggregate = run_baseline_evaluation(
        subjects=tuple(args.subjects),
        runs=tuple(args.runs),
        data_dir=Path(args.data_dir),
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        apply_notch=args.notch,
        csp_components=args.csp_components,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_path = out_dir / "baseline_per_fold.csv"
    aggregate_path = out_dir / "baseline_aggregate.csv"
    per_fold.to_csv(per_fold_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    print(f"Wrote per-fold results to {per_fold_path}")
    print(f"Wrote aggregate results to {aggregate_path}")
    print(aggregate.to_string(index=False))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.preprocessing_sweep import run_preprocessing_sweep

    results = run_preprocessing_sweep(
        subjects=tuple(args.subjects), runs=tuple(args.runs), data_dir=Path(args.data_dir)
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print(f"Wrote preprocessing sweep results to {out_path}")
    real = results[~results["shuffled_control"]]
    print(
        real[
            [
                "preprocessing_candidate",
                "model",
                "mean_balanced_accuracy",
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
            ]
        ].to_string(index=False)
    )
    return 0


def cmd_select_preprocessing(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.preprocessing_selection import select_preprocessing_candidate

    result = select_preprocessing_candidate(
        subjects=tuple(args.subjects),
        runs=tuple(args.runs),
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
    )
    print(f"Chosen preprocessing candidate: {result['chosen_candidate']}")
    print(f"Selected using validation subjects: {result['selection_subjects']}")
    print("Candidate scores (mean balanced accuracy):")
    for name, score in result["candidate_scores"].items():
        print(f"  {name}: {score:.4f}")
    print(f"Wrote frozen selection to {args.output}")
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from calm.mae.pretrain import pretrain_mae

    kwargs = dict(
        subjects=tuple(args.subjects),
        runs=tuple(args.runs),
        data_dir=Path(args.data_dir),
        n_train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=Path(args.checkpoint),
        mask_strategy=args.mask_strategy,
    )
    selection_path = Path(args.selection)
    if selection_path.exists():
        params = json.loads(selection_path.read_text())["candidate_params"]
        epoch_cfg = params["epoch_config"]
        from calm.signal.epochs import EpochConfig

        kwargs.update(
            l_freq=params["l_freq"], h_freq=params["h_freq"], apply_notch=params["apply_notch"],
            apply_reference=params["apply_reference"],
            epoch_config=EpochConfig(tmin=epoch_cfg["tmin"], tmax=epoch_cfg["tmax"]),
        )
        print(f"Using frozen preprocessing candidate '{params['name']}' from {selection_path}")
    else:
        print(f"No frozen preprocessing selection at {selection_path}; using defaults")

    result = pretrain_mae(**kwargs)
    print(f"Pretrained on {len(args.subjects)} dev subjects, {result['n_samples']} epochs")
    print(f"Final masked-reconstruction loss: {result['loss_history'][-1]:.4f}")
    print(f"Loss history: {[round(v, 4) for v in result['loss_history']]}")
    print(f"Wrote checkpoint to {result['checkpoint_path']}")
    return 0


def cmd_evaluate_held_out(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.held_out_evaluation import evaluate_held_out

    per_fold, per_subject, aggregate = evaluate_held_out(
        subjects=tuple(args.subjects),
        runs=tuple(args.runs),
        data_dir=Path(args.data_dir),
        selection_path=Path(args.selection),
        mae_checkpoint_path=Path(args.mae_checkpoint) if args.mae_checkpoint else None,
        output_dir=Path(args.output_dir),
    )
    print(f"Held-out evaluation on {sorted(set(args.subjects))} (cross-run evaluation)")
    print("\nPer-subject results:")
    print(per_subject.to_string(index=False))
    print("\nAggregate results (bootstrap CI over subjects):")
    print(aggregate.to_string(index=False))
    print(f"\nWrote results to {args.output_dir}/held_out_{{per_fold,per_subject,aggregate}}.csv")
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.robustness.evaluate_robustness import evaluate_channel_dropout_robustness

    encoder_checkpoints = {}
    default_mae = ARTIFACTS_DIR / "checkpoints" / "mae.pt"
    channel_mae = ARTIFACTS_DIR / "checkpoints" / "mae_channel.pt"
    if default_mae.exists():
        encoder_checkpoints["random_mask_pretrained"] = default_mae
    if channel_mae.exists():
        encoder_checkpoints["channel_mask_pretrained"] = channel_mae

    results = evaluate_channel_dropout_robustness(
        subjects=tuple(args.subjects),
        runs=tuple(args.runs),
        data_dir=Path(args.data_dir),
        selection_path=Path(args.selection),
        encoder_checkpoints=encoder_checkpoints,
        dropout_fraction=args.dropout_fraction,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    summary = results.groupby("model")[
        ["clean_balanced_accuracy", "corrupted_balanced_accuracy", "degradation"]
    ].mean()
    print(f"Wrote {len(results)} rows to {out_path}")
    print(f"\n{args.dropout_fraction:.0%} random channel dropout, mean over subjects/folds:")
    print(summary.to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calm", description="CALM EEG experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="Download PhysioNet EEGBCI subjects/runs")
    _add_subjects_runs_args(p_download)
    p_download.set_defaults(func=cmd_download)

    p_audit = subparsers.add_parser("audit", help="Build the subject-run audit manifest")
    _add_subjects_runs_args(p_audit)
    p_audit.add_argument(
        "--output", type=str, default=str(RESULTS_DIR / "audit_manifest.csv"),
        help="Manifest CSV output path (default: %(default)s)",
    )
    p_audit.set_defaults(func=cmd_audit)

    p_explore = subparsers.add_parser(
        "explore", help="Generate exploratory figures for one subject"
    )
    p_explore.add_argument("--subject", type=int, default=1)
    p_explore.add_argument(
        "--runs", type=int, nargs="+", default=list(LEFT_RIGHT_RUNS),
        help="Left/right motor-imagery runs (default: %(default)s)",
    )
    p_explore.add_argument("--data-dir", type=str, default=str(DATA_RAW_DIR))
    p_explore.set_defaults(func=cmd_explore)

    p_baseline = subparsers.add_parser(
        "baseline", help="Run classical baselines with leakage-free cross-run evaluation"
    )
    _add_subjects_runs_args(p_baseline)
    p_baseline.add_argument("--l-freq", type=float, default=4.0, help="Band-pass low edge (Hz)")
    p_baseline.add_argument("--h-freq", type=float, default=40.0, help="Band-pass high edge (Hz)")
    p_baseline.add_argument("--notch", action="store_true", help="Apply a 60 Hz notch filter")
    p_baseline.add_argument("--csp-components", type=int, default=6)
    p_baseline.add_argument(
        "--output-dir", type=str, default=str(RESULTS_DIR),
        help="Directory for baseline_per_fold.csv / baseline_aggregate.csv (default: %(default)s)",
    )
    p_baseline.set_defaults(func=cmd_baseline)

    p_sweep = subparsers.add_parser(
        "sweep", help="Compare preprocessing candidates (reference/band/notch/epoch window)"
    )
    _add_subjects_runs_args(p_sweep)
    p_sweep.add_argument(
        "--output", type=str, default=str(RESULTS_DIR / "preprocessing_sweep.csv"),
        help="Sweep results CSV output path (default: %(default)s)",
    )
    p_sweep.set_defaults(func=cmd_sweep)

    p_select = subparsers.add_parser(
        "select-preprocessing",
        help="Freeze a preprocessing candidate using validation subjects only (37-45)",
    )
    p_select.add_argument(
        "--subjects", type=int, nargs="+", default=list(VALIDATION_SUBJECTS),
        help="Validation subjects only (default: %(default)s)",
    )
    p_select.add_argument(
        "--runs", type=int, nargs="+", default=list(LEFT_RIGHT_RUNS),
        help="Left/right motor-imagery runs (default: %(default)s)",
    )
    p_select.add_argument("--data-dir", type=str, default=str(DATA_RAW_DIR))
    p_select.add_argument(
        "--output", type=str, default=str(RESULTS_DIR / "preprocessing_selection.json"),
        help="Frozen-selection JSON output path (default: %(default)s)",
    )
    p_select.set_defaults(func=cmd_select_preprocessing)

    p_pretrain = subparsers.add_parser(
        "pretrain", help="Pretrain the masked autoencoder on development subjects (1-36)"
    )
    p_pretrain.add_argument(
        "--subjects", type=int, nargs="+", default=list(DEV_SUBJECTS),
        help="Development subjects only (default: %(default)s)",
    )
    p_pretrain.add_argument(
        "--runs", type=int, nargs="+", default=list(LEFT_RIGHT_RUNS),
        help="Left/right motor-imagery runs (default: %(default)s)",
    )
    p_pretrain.add_argument("--data-dir", type=str, default=str(DATA_RAW_DIR))
    p_pretrain.add_argument(
        "--selection", type=str, default=str(RESULTS_DIR / "preprocessing_selection.json"),
        help="Frozen preprocessing-selection JSON to use if present (default: %(default)s)",
    )
    p_pretrain.add_argument("--train-epochs", type=int, default=10)
    p_pretrain.add_argument("--batch-size", type=int, default=32)
    p_pretrain.add_argument("--lr", type=float, default=1e-3)
    p_pretrain.add_argument(
        "--mask-strategy", type=str, default="random", choices=["random", "whole_channel"],
        help="Token masking strategy (default: %(default)s)",
    )
    p_pretrain.add_argument(
        "--checkpoint", type=str, default=str(ARTIFACTS_DIR / "checkpoints" / "mae.pt"),
        help="Checkpoint output path (default: %(default)s)",
    )
    p_pretrain.set_defaults(func=cmd_pretrain)

    p_eval = subparsers.add_parser(
        "evaluate-held-out",
        help="Cross-run evaluation on subjects 46-60 using saved preprocessing",
    )
    p_eval.add_argument(
        "--subjects", type=int, nargs="+", default=list(TEST_SUBJECTS),
        help="Held-out test subjects only (default: %(default)s)",
    )
    p_eval.add_argument(
        "--runs", type=int, nargs="+", default=list(LEFT_RIGHT_RUNS),
        help="Left/right motor-imagery runs (default: %(default)s)",
    )
    p_eval.add_argument("--data-dir", type=str, default=str(DATA_RAW_DIR))
    p_eval.add_argument(
        "--selection", type=str, default=str(RESULTS_DIR / "preprocessing_selection.json"),
        help="Frozen preprocessing-selection JSON to use (default: %(default)s)",
    )
    p_eval.add_argument(
        "--mae-checkpoint", type=str, default=str(ARTIFACTS_DIR / "checkpoints" / "mae.pt"),
        help="Pretrained MAE checkpoint to include as frozen-encoder probes; "
        "pass empty string to skip (default: %(default)s)",
    )
    p_eval.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    p_eval.set_defaults(func=cmd_evaluate_held_out)

    p_robust = subparsers.add_parser(
        "robustness",
        help="Compare clean vs. random-channel-dropout performance on held-out subjects",
    )
    p_robust.add_argument(
        "--subjects", type=int, nargs="+", default=list(TEST_SUBJECTS),
        help="Held-out test subjects only (default: %(default)s)",
    )
    p_robust.add_argument(
        "--runs", type=int, nargs="+", default=list(LEFT_RIGHT_RUNS),
        help="Left/right motor-imagery runs (default: %(default)s)",
    )
    p_robust.add_argument("--data-dir", type=str, default=str(DATA_RAW_DIR))
    p_robust.add_argument(
        "--selection", type=str, default=str(RESULTS_DIR / "preprocessing_selection.json"),
    )
    p_robust.add_argument("--dropout-fraction", type=float, default=0.25)
    p_robust.add_argument(
        "--output", type=str, default=str(RESULTS_DIR / "robustness_channel_dropout.csv"),
    )
    p_robust.set_defaults(func=cmd_robustness)

    p_calibration = subparsers.add_parser(
        "calibration", help="Run exploratory paired calibration curves with fixed checkpoints",
    )
    p_calibration.add_argument("--config", default="configs/calibration.yaml")
    p_calibration.add_argument("--output-dir", default=None)
    p_calibration.set_defaults(func=cmd_calibration)

    return parser


def cmd_calibration(args: argparse.Namespace) -> int:
    from pathlib import Path

    from calm.calibration import run_calibration

    output = run_calibration(
        Path(args.config), Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Calibration results: {output / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
