"""Raw traces, montage, spectra, and cue-aligned power plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never requires a display

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci

from calm.config import DATA_RAW_DIR, FIGURES_DIR, LEFT_RIGHT_RUNS
from calm.data.download import download_eegbci_subjects
from calm.data.events import LEFT_HAND_LABEL, RIGHT_HAND_LABEL, map_left_right_annotations

logger = logging.getLogger(__name__)

EPOCH_TMIN, EPOCH_TMAX = -1.0, 4.0
TFR_FREQS = np.arange(4.0, 40.0, 1.0)
MOTOR_CHANNELS = ("C3", "C4", "Cz")


def load_subject_left_right_raw(
    subject: int,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
) -> mne.io.BaseRaw:
    """Load and concatenate a subject's left/right runs with mapped annotations."""
    subject_paths = download_eegbci_subjects(subjects=(subject,), runs=runs, data_dir=data_dir)
    raws = []
    for path in sorted(subject_paths[subject]):
        run = int(path.stem.split("R")[-1])
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw = map_left_right_annotations(raw, run)
        raws.append(raw)
    concatenated = mne.concatenate_raws(raws)
    eegbci.standardize(concatenated)  # renames e.g. "Fc5." -> "FC5" for montage lookup
    montage = mne.channels.make_standard_montage("colin27_1005")
    concatenated.set_montage(montage, on_missing="warn")
    return concatenated


def _savefig(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


def plot_raw_traces(raw: mne.io.BaseRaw, out_dir: Path) -> Path:
    fig = raw.plot(duration=10, n_channels=20, show=False)
    return _savefig(fig, out_dir, "01_raw_traces.png")


def plot_montage(raw: mne.io.BaseRaw, out_dir: Path) -> Path:
    fig = raw.plot_sensors(show_names=True, show=False)
    return _savefig(fig, out_dir, "02_montage.png")


def plot_psd(raw: mne.io.BaseRaw, out_dir: Path) -> Path:
    psd = raw.compute_psd(fmin=1, fmax=80, verbose=False)
    fig = psd.plot(show=False)
    return _savefig(fig, out_dir, "03_psd.png")


def plot_60hz_inspection(raw: mne.io.BaseRaw, out_dir: Path) -> Path:
    psd = raw.compute_psd(fmin=45, fmax=75, verbose=False)
    freqs = psd.freqs
    power_db = 10 * np.log10(psd.get_data().mean(axis=0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, power_db)
    ax.axvline(60.0, color="red", linestyle="--", label="60 Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB, mean across channels)")
    ax.set_title("Line-noise (60 Hz) inspection")
    ax.legend()
    return _savefig(fig, out_dir, "04_60hz_inspection.png")


def plot_cue_aligned_tfr(raw: mne.io.BaseRaw, out_dir: Path) -> list[Path]:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    wanted = {k: v for k, v in event_id.items() if k in (LEFT_HAND_LABEL, RIGHT_HAND_LABEL)}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )
    channels = [ch for ch in MOTOR_CHANNELS if ch in epochs.ch_names]
    if not channels:
        channels = epochs.ch_names[:1]

    paths = []
    for label in (LEFT_HAND_LABEL, RIGHT_HAND_LABEL):
        if label not in epochs.event_id:
            continue
        cond_epochs = epochs[label]
        tfr = cond_epochs.compute_tfr(
            method="multitaper",
            freqs=TFR_FREQS,
            n_cycles=TFR_FREQS / 2.0,
            average=True,
            return_itc=False,
            picks=channels,
            verbose=False,
        )
        fig = tfr.plot(picks=channels, combine="mean", show=False)
        if isinstance(fig, list):
            fig = fig[0]
        fig.suptitle(f"Cue-aligned time-frequency power: {label}")
        paths.append(_savefig(fig, out_dir, f"05_tfr_{label.lower()}.png"))
    return paths


def generate_subject_figures(
    subject: int = 1,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
    out_dir: Path | None = None,
) -> list[Path]:
    out_dir = out_dir or (FIGURES_DIR / f"subject_{subject:02d}")
    raw = load_subject_left_right_raw(subject, runs=runs, data_dir=data_dir)

    saved = [
        plot_raw_traces(raw, out_dir),
        plot_montage(raw, out_dir),
        plot_psd(raw, out_dir),
        plot_60hz_inspection(raw, out_dir),
    ]
    saved.extend(plot_cue_aligned_tfr(raw, out_dir))
    return saved
