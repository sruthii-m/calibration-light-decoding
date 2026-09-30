"""Train the masked autoencoder on development subjects."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from calm.config import ARTIFACTS_DIR, DATA_RAW_DIR, DEV_SUBJECTS, LEFT_RIGHT_RUNS
from calm.data.download import download_eegbci_subjects
from calm.mae.model import MAEConfig, MaskedAutoencoder
from calm.representation.stft_frontend import STFTConfig, TrainOnlyNormalizer, epochs_to_log_power
from calm.representation.tokens import DEFAULT_BANDS, TokenGeometry, pool_to_band_time_tokens
from calm.signal.epochs import EpochConfig, left_right_epochs_from_raw
from calm.signal.filtering import bandpass_filter_offline, common_average_reference, notch_filter

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = ARTIFACTS_DIR / "checkpoints" / "mae.pt"


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_pretraining_tokens(
    subjects: tuple[int, ...],
    runs: tuple[int, ...],
    data_dir: Path,
    l_freq: float,
    h_freq: float,
    apply_notch: bool,
    apply_reference: bool,
    epoch_config: EpochConfig,
    stft_config: STFTConfig,
    n_time_bins: int,
    bands: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, list[str]]:
    import mne

    all_X: list[np.ndarray] = []
    ch_names: list[str] | None = None
    for subject in subjects:
        paths = download_eegbci_subjects(subjects=(subject,), runs=runs, data_dir=data_dir)
        for path in sorted(paths[subject]):
            run = int(path.stem.split("R")[-1])
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            if apply_reference:
                raw = common_average_reference(raw)
            raw = bandpass_filter_offline(raw, l_freq, h_freq)
            if apply_notch:
                raw = notch_filter(raw)
            # Rest epochs add unlabeled data but are excluded from classification.
            epochs = left_right_epochs_from_raw(raw, run, config=epoch_config, include_rest=True)
            all_X.append(epochs.get_data(copy=True))
            logger.info("subject=%d run=%d pretraining_epochs=%d", subject, run, len(epochs))
            if ch_names is None:
                ch_names = list(raw.ch_names)

    X_all = np.concatenate(all_X, axis=0)
    log_power, freqs, _times = epochs_to_log_power(X_all, sfreq=160.0, config=stft_config)
    tokens = pool_to_band_time_tokens(log_power, freqs, bands=bands, n_time_bins=n_time_bins)
    return tokens, ch_names


def pretrain_mae(
    subjects: tuple[int, ...] = DEV_SUBJECTS,
    runs: tuple[int, ...] = LEFT_RIGHT_RUNS,
    data_dir: Path = DATA_RAW_DIR,
    mae_config: MAEConfig | None = None,
    n_time_bins: int = 4,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
    l_freq: float = 4.0,
    h_freq: float = 40.0,
    apply_notch: bool = False,
    apply_reference: bool = True,
    epoch_config: EpochConfig | None = None,
    n_train_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    mask_strategy: str = "random",
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> dict:
    subjects_set = set(subjects)
    if not subjects_set.issubset(DEV_SUBJECTS):
        raise ValueError(
            f"MAE pretraining must use only development subjects {DEV_SUBJECTS}; "
            f"got {sorted(subjects_set)}. Pretraining on validation/held-out "
            "subjects would leak them into the encoder before evaluation."
        )

    mae_config = mae_config or MAEConfig()
    epoch_config = epoch_config or EpochConfig()
    stft_config = STFTConfig(fmax=h_freq)

    logger.info("Building pretraining tokens from dev subjects %s", subjects)
    tokens, ch_names = build_pretraining_tokens(
        subjects, runs, data_dir, l_freq, h_freq, apply_notch, apply_reference,
        epoch_config, stft_config, n_time_bins, bands,
    )
    logger.info("Pretraining token tensor shape: %s", tokens.shape)

    normalizer = TrainOnlyNormalizer().fit(tokens)
    normalized = normalizer.transform(tokens).astype(np.float32)

    geometry = TokenGeometry.from_channels(ch_names, n_time_bins=n_time_bins, bands=bands)
    device = select_device()
    model = MaskedAutoencoder(geometry, mae_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = torch.from_numpy(normalized)
    n_samples = data.shape[0]
    rng = np.random.default_rng(seed)

    loss_history: list[float] = []
    for epoch in range(n_train_epochs):
        order = rng.permutation(n_samples)
        epoch_losses = []
        for step, start in enumerate(range(0, n_samples, batch_size)):
            batch_idx = order[start : start + batch_size]
            batch = data[batch_idx].to(device)

            optimizer.zero_grad()
            loss = model.forward_loss(
                batch, seed=epoch * 100_000 + step, mask_strategy=mask_strategy
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)
        logger.info("pretrain epoch %d/%d: mean masked-reconstruction loss=%.4f",
                     epoch + 1, n_train_epochs, mean_loss)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.to("cpu").state_dict(),
            "mae_config": asdict(mae_config),
            "geometry": {
                "ch_names": geometry.ch_names,
                "n_bands": geometry.n_bands,
                "n_time_bins": geometry.n_time_bins,
                "bands": geometry.bands,
            },
            "normalizer_mean": normalizer.mean_,
            "normalizer_std": normalizer.std_,
            "stft_config": asdict(stft_config),
            "preprocessing": {
                "l_freq": l_freq, "h_freq": h_freq, "apply_notch": apply_notch,
                "apply_reference": apply_reference,
                "epoch_config": asdict(epoch_config),
            },
            "pretraining_subjects": sorted(subjects_set),
            "loss_history": loss_history,
            "mask_strategy": mask_strategy,
        },
        checkpoint_path,
    )
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"loss_history": loss_history, "n_samples": n_samples,
                                          "pretraining_subjects": sorted(subjects_set)}, indent=2))
    logger.info("Saved MAE checkpoint to %s", checkpoint_path)

    return {
        "loss_history": loss_history,
        "n_samples": n_samples,
        "checkpoint_path": checkpoint_path,
    }
