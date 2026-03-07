#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:10:04 2026

@author: mac
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from mne.time_frequency import psd_array_welch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score
)

from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import FastICA


# ----------------------------
# Basic parameters
# ----------------------------
LOW = 1
HIGH = 60
FS = 125   # OpenBCI Daisy sampling rate

LABEL_MAP = {
    "sad": 0,
    "happy": 1
}

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)   # safer than going all the way to 60
}


# ----------------------------
# Load CSV data
# ----------------------------
def load_data(filepath):
    df = pd.read_csv(
        filepath,
        comment='%',
        sep=None,
        engine='python'
    )

    print(f"\nLoaded file: {filepath}")
    print(df.head())

    # Assuming EEG is stored in columns 1:17
    eeg = df.iloc[:, 1:17].values.T   # shape: (channels, samples)

    # Convert to microvolts if your scaling is correct
    eeg *= 0.02235

    # Remove per-channel DC offset
    eeg -= np.mean(eeg, axis=1, keepdims=True)

    return eeg


# ----------------------------
# Filtering
# ----------------------------
def bandpass_filter(data, low, high, fs, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, data, axis=1)

def notch_filter(data, freq, fs, quality=30):
    b, a = iirnotch(freq / (fs / 2), quality)
    return filtfilt(b, a, data, axis=1)

def clean_eeg(eeg):
    eeg = bandpass_filter(eeg, LOW, HIGH, FS, order=4)
    eeg = notch_filter(eeg, 60, FS, quality=30)

    # ICA block (currently no components are removed unless you specify them)
    ica = FastICA(n_components=16, random_state=42, max_iter=1000)
    source_estimated = ica.fit_transform(eeg.T)   # (samples, channels)
    sources_clean = source_estimated.copy()

    # Add component indices here if you identify noisy ICs
    noise = []
    if len(noise) > 0:
        sources_clean[:, noise] = 0

    eeg_clean = ica.inverse_transform(sources_clean).T
    eeg_clean -= np.mean(eeg_clean, axis=1, keepdims=True)

    return eeg_clean


# ----------------------------
# Epoching
# ----------------------------
def epoch_data(eeg, epoch_length, fs):
    samples_per_epoch = int(epoch_length * fs)
    num_epochs = eeg.shape[1] // samples_per_epoch

    eeg = eeg[:, :num_epochs * samples_per_epoch]
    eeg = eeg.reshape(eeg.shape[0], num_epochs, samples_per_epoch)

    return eeg   # shape: (channels, epochs, samples_per_epoch)


# ----------------------------
# Bandpower extraction
# ----------------------------
def bandpower(epoch, fs, band, n_fft=256):
    n_times = epoch.shape[-1]
    n_fft = min(n_fft, n_times)

    psd, freqs = psd_array_welch(epoch, sfreq=fs, n_fft=n_fft, verbose=False)

    low, high = band
    idx_band = (freqs >= low) & (freqs <= high)
    band_psd = psd[..., idx_band]

    return np.mean(band_psd, axis=-1)   # one value per channel


# ----------------------------
# Feature extraction
# ----------------------------
def av_extract(eeg, epoch_length, fs):
    eeg_epoch = epoch_data(eeg, epoch_length, fs)
    eeg_epoch = eeg_epoch.transpose(1, 0, 2)   # (epochs, channels, samples)

    feature_matrix = []

    for epoch in eeg_epoch:
        delta_power = bandpower(epoch, fs, BANDS['delta'])
        theta_power = bandpower(epoch, fs, BANDS['theta'])
        alpha_power = bandpower(epoch, fs, BANDS['alpha'])
        beta_power  = bandpower(epoch, fs, BANDS['beta'])
        gamma_power = bandpower(epoch, fs, BANDS['gamma'])

        alpha_beta_ratio = alpha_power / (beta_power + 1e-8)

        features = np.concatenate([
            delta_power,
            theta_power,
            alpha_power,
            beta_power,
            gamma_power,
            alpha_beta_ratio
        ])

        feature_matrix.append(features)

    return np.array(feature_matrix)   # shape: (n_epochs, n_features)


# ----------------------------
# One file -> X, y
# Each epoch inherits the file label
# ----------------------------
def extract_features_and_labels(filepath, label, epoch_length=2.0, fs=FS):
    eeg = load_data(filepath)
    eeg = clean_eeg(eeg)

    X_file = av_extract(eeg, epoch_length, fs)
    y_file = np.full(X_file.shape[0], label)

    print(f"Extracted features from {filepath}")
    print(f"X_file shape: {X_file.shape}, y_file shape: {y_file.shape}")

    return X_file, y_file


# ----------------------------
# Multiple files -> full dataset
# file_label_list format:
# [(filepath1, label1), (filepath2, label2), ...]
# ----------------------------
def build_dataset(file_label_list, epoch_length=2.0, fs=FS):
    X_all = []
    y_all = []

    for filepath, label in file_label_list:
        print(f"\nProcessing file: {filepath} | label = {label}")
        X_file, y_file = extract_features_and_labels(filepath, label, epoch_length, fs)
        X_all.append(X_file)
        y_all.append(y_file)

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print("\nFinal dataset summary:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class counts: {np.bincount(y)}")

    return X, y


# ----------------------------
# SVM training and evaluation
# ----------------------------
def train_svm_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC())
    ])

    param_grid = {
        "svm__kernel": ["linear", "rbf"],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.1, 1]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid.best_params_)
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["sad", "happy"]
    ))

    ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        display_labels=["sad", "happy"]
    )
    plt.title("Happy vs Sad Confusion Matrix")
    plt.show()

    return best_model, grid


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    file_label_list = [
        ("OpenBCI_Data/2-7/OpenBCISession_fail_happy_stimuli_2_7/BrainFlow-RAW_happy_stimuli_2_7_0.csv", LABEL_MAP["happy"]),
        ("OpenBCI_Data/2-7/OpenBCISession_fail_sad_stimuli_2_7/BrainFlow-RAW_sad_stimuli_2_7_0.csv", LABEL_MAP["sad"]),
    ]

    X, y = build_dataset(file_label_list, epoch_length=2.0, fs=FS)
    best_model, grid = train_svm_classifier(X, y)