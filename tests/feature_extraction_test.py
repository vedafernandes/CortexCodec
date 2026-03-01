import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roughccpipeline import (
    load_data,
    bandpass_filter,
    notch_filter,
    clean_eeg,
    epoch_data,
    bandpower,
    av_extract,
)


def test_load_data_shapes_and_centering(tmp_path):
    n_samples = 10
    data = {"time": np.arange(n_samples)}
    for i in range(1, 17):
        data[f"ch{i}"] = np.linspace(0, 1, n_samples)
    df = pd.DataFrame(data)
    csv_path = tmp_path / "eeg.csv"
    df.to_csv(csv_path, index=False)

    eeg = load_data(str(csv_path))

    assert eeg.shape == (16, n_samples)
    assert np.allclose(eeg.mean(axis=1), 0.0, atol=1e-6)


def test_bandpass_filter_removes_dc_component():
    channels = 2
    n_samples = 1000
    data = np.ones((channels, n_samples))

    filtered = bandpass_filter(data, low=1, high=60, fs=125, order=4)

    assert filtered.shape == data.shape
    assert np.allclose(filtered.mean(axis=1), 0.0, atol=1e-2)


def test_notch_filter_at_60_hz():
    fs = 125
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    signal = np.sin(2 * np.pi * 60 * t)
    data = signal[np.newaxis, :]

    filtered = notch_filter(data, freq=60, fs=fs, quality=30)

    orig_rms = np.sqrt(np.mean(data**2))
    filtered_rms = np.sqrt(np.mean(filtered**2))
    assert filtered_rms < orig_rms * 0.7


def test_clean_eeg_output_shape_and_centering():
    rng = np.random.RandomState(0)
    channels = 16
    n_samples = 1000
    eeg = rng.randn(channels, n_samples)

    eeg_clean = clean_eeg(eeg)

    assert eeg_clean.shape == eeg.shape
    assert np.allclose(eeg_clean.mean(axis=1), 0.0, atol=1e-6)


def test_epoch_data_shapes():
    channels = 16
    fs = 125
    epoch_length = 2.0
    n_epochs = 3
    n_samples = int(fs * epoch_length * n_epochs)
    eeg = np.random.randn(channels, n_samples)

    epochs = epoch_data(eeg, epoch_length=epoch_length, fs=fs)

    assert epochs.shape == (channels, n_epochs, int(fs * epoch_length))


def test_bandpower_higher_in_matching_band():
    fs = 125
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    signal = np.sin(2 * np.pi * 10 * t)
    epoch = signal[np.newaxis, :]

    alpha_power = bandpower(epoch, fs=fs, band=(8, 13))
    beta_power = bandpower(epoch, fs=fs, band=(13, 30))

    assert alpha_power.shape == beta_power.shape == (1,)
    assert alpha_power[0] > beta_power[0]


def test_av_extract_feature_matrix_shape():
    rng = np.random.RandomState(1)
    channels = 16
    fs = 125
    epoch_length = 1.0
    n_epochs = 4
    n_samples = int(fs * epoch_length * n_epochs)
    eeg = rng.randn(channels, n_samples)

    features = av_extract(eeg, epoch_length=epoch_length, fs=fs)

    assert features.shape == (n_epochs, channels * 3)
