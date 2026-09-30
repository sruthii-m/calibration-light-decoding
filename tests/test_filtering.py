from __future__ import annotations

import numpy as np

from calm.signal.filtering import bandpass_filter_offline, causal_bandpass, common_average_reference


def test_bandpass_recovers_spectral_peak(synthetic_raw_run4):
    sfreq = synthetic_raw_run4.info["sfreq"]
    n_times = synthetic_raw_run4.n_times
    t = np.arange(n_times) / sfreq

    raw = synthetic_raw_run4.copy()
    data = raw.get_data()
    # Inject a strong in-band (10 Hz) signal and a strong out-of-band (70 Hz) signal.
    data[0, :] += 5e-6 * np.sin(2 * np.pi * 10 * t) + 5e-6 * np.sin(2 * np.pi * 70 * t)
    raw._data[:] = data

    filtered = bandpass_filter_offline(raw, l_freq=4.0, h_freq=40.0)

    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(filtered.get_data()[0]))
    power_at = lambda f: spectrum[np.argmin(np.abs(freqs - f))]  # noqa: E731

    assert power_at(10.0) > 10 * power_at(70.0)


def test_common_average_reference_zeros_channel_mean(synthetic_raw_run4):
    referenced = common_average_reference(synthetic_raw_run4)
    mean_across_channels = referenced.get_data().mean(axis=0)
    np.testing.assert_allclose(mean_across_channels, 0.0, atol=1e-12)


def test_bandpass_filter_offline_does_not_mutate_input(synthetic_raw_run4):
    original = synthetic_raw_run4.get_data().copy()
    bandpass_filter_offline(synthetic_raw_run4, l_freq=4.0, h_freq=40.0)
    np.testing.assert_array_equal(synthetic_raw_run4.get_data(), original)


def test_causal_filter_has_no_future_sample_access():
    sfreq = 160.0
    n_times = 800
    rng = np.random.default_rng(1)
    t = np.arange(n_times) / sfreq
    base = np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(n_times)

    split = 400
    modified = base.copy()
    modified[split:] += 50.0  # drastically alter only future samples

    filtered_base = causal_bandpass(base, sfreq, l_freq=4.0, h_freq=40.0)
    filtered_modified = causal_bandpass(modified, sfreq, l_freq=4.0, h_freq=40.0)

    # Samples before the perturbation must be unchanged.
    np.testing.assert_allclose(filtered_base[:split], filtered_modified[:split])


def test_offline_zero_phase_filter_can_depend_on_future_samples(synthetic_raw_run4):
    """A nearby future perturbation can affect zero-phase FIR output."""
    raw_a = synthetic_raw_run4.copy()
    raw_b = synthetic_raw_run4.copy()
    data_b = raw_b.get_data()
    data_b[:, 20:40] += 50.0  # perturb samples shortly after t=0
    raw_b._data[:] = data_b

    filtered_a = bandpass_filter_offline(raw_a, l_freq=4.0, h_freq=40.0).get_data()
    filtered_b = bandpass_filter_offline(raw_b, l_freq=4.0, h_freq=40.0).get_data()

    # Zero-phase filtering is non-causal: an early sample's output CAN change
    # when a nearby later sample changes.
    assert not np.allclose(filtered_a[:, :10], filtered_b[:, :10])
