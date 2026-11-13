"""
Human Activity Recognition from smartphone inertial signals (UCI HAR).

Pipeline:
    raw windows (N, 128, 9) -> 40 engineered features -> StandardScaler
    -> classifier -> activity

Sections printed by this script:
    1. Data loading
    2. Unscaled baseline (18 features)
    3. Model comparison (40 features, scaled)
    4. Feature ablation (Logistic Regression, scaled)
    5. Per-class report + normalized confusion matrix (Gradient Boosting)
"""

import os

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

SEED = 42
FS = 50  # sampling rate in Hz
BASE_DIR = "UCI HAR Dataset/UCI HAR Dataset/"
RESULTS_DIR = "results"

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
CLASS_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_split_3d(dataset_path, split):
    """Load the 9 inertial channels of a split into an array (N, 128, 9)."""
    channels = []
    for sig in SIGNALS:
        path = f"{dataset_path}{split}/Inertial Signals/{sig}_{split}.txt"
        channels.append(pd.read_csv(path, sep=r"\s+", header=None).values)
    return np.stack(channels, axis=-1)


def load_labels(dataset_path, split):
    path = f"{dataset_path}{split}/y_{split}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None).values.ravel()


# --------------------------------------------------------------------------
# 2. Feature extraction (groups kept separate so the ablation can reuse them)
# --------------------------------------------------------------------------
def feat_mean_std(X):
    """Mean and std per channel -> (N, 18)."""
    return np.hstack([X.mean(axis=1), X.std(axis=1)])


def feat_magnitude(X):
    """Mean and std of the acceleration magnitude -> (N, 4).

    Order: body_mean, total_mean, body_std, total_std.
    """
    body_mag = np.linalg.norm(X[:, :, 0:3], axis=2)    # (N, 128)
    total_mag = np.linalg.norm(X[:, :, 6:9], axis=2)   # (N, 128)
    return np.column_stack([
        body_mag.mean(axis=1), total_mag.mean(axis=1),
        body_mag.std(axis=1), total_mag.std(axis=1),
    ])


def _spectrum(X):
    """One-sided FFT magnitude of each channel with the DC component removed."""
    X_centered = X - X.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(X_centered, axis=1))      # (N, 65, 9)
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / FS)    # (65,)
    return mag, freqs


def feat_dominant_freq(X):
    """Frequency (Hz) of the strongest non-DC FFT bin per channel -> (N, 9)."""
    mag, freqs = _spectrum(X)
    idx = np.argmax(mag[:, 1:, :], axis=1) + 1          # skip bin 0 (DC)
    return freqs[idx]


def feat_spectral_energy(X):
    """Signal energy from the FFT (Parseval, DC removed) per channel -> (N, 9)."""
    mag, _ = _spectrum(X)
    return np.sum(mag ** 2, axis=1) / X.shape[1]


def extract_features(X):
    """All 40 features, ordered by ablation group: 18 + 4 + 9 + 9."""
    return np.hstack([
        feat_mean_std(X),
        feat_magnitude(X),
        feat_dominant_freq(X),
        feat_spectral_energy(X),
    ])


# Column counts after adding each group (for the ablation table)
ABLATION_STEPS = [
    ("Mean + std", 18),
    ("+ Magnitude", 22),
    ("+ Dominant frequency", 31),
    ("+ Spectral energy", 40),
]


def scaled_logreg_accuracy(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler().fit(X_tr)  # fit on train only: no leakage
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(scaler.transform(X_tr), y_tr)
    return accuracy_score(y_te, clf.predict(scaler.transform(X_te)))


# --------------------------------------------------------------------------
# Prediction on a single window (used in the README)
# --------------------------------------------------------------------------
def predict_window(window, scaler, model):
    """Predict the activity of one window of shape (128, 9)."""
    window = np.asarray(window, dtype=float)
    if window.shape != (128, 9):
        raise ValueError(f"Expected shape (128, 9), got {window.shape}")
    feats = extract_features(window[np.newaxis])
    label = model.predict(scaler.transform(feats))[0]
    return CLASS_NAMES[int(label) - 1]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading signal data...")
    X_train = load_split_3d(BASE_DIR, "train")  # (7352, 128, 9)
    X_test = load_split_3d(BASE_DIR, "test")    # (2947, 128, 9)
    y_train = load_labels(BASE_DIR, "train")
    y_test = load_labels(BASE_DIR, "test")
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")

    F_train = extract_features(X_train)
    F_test = extract_features(X_test)
    print(f"Features: train {F_train.shape}, test {F_test.shape}")

    # Scaler fitted on the training set only
    scaler = StandardScaler().fit(F_train)
    F_train_s = scaler.transform(F_train)
    F_test_s = scaler.transform(F_test)

    # ---- Unscaled baseline (18 features) ---------------------------------
    base = LogisticRegression(max_iter=1000, random_state=SEED)
    base.fit(F_train[:, :18], y_train)
    base_acc = accuracy_score(y_test, base.predict(F_test[:, :18]))
    print(f"\nUnscaled baseline (mean + std, 18 features): {base_acc * 100:.2f}%")

    # ---- Model comparison -------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=SEED
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, random_state=SEED
        ),
    }

    print("\n" + "=" * 55)
    print("         MODEL COMPARISON ON 40 DSP FEATURES")
    print("=" * 55)

    results = {}
    predictions = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(F_train_s, y_train)
        predictions[name] = model.predict(F_test_s)
        results[name] = accuracy_score(y_test, predictions[name])

    print("\n" + "-" * 45)
    print(f"{'Classifier Model':<25} | {'Test Accuracy':<15}")
    print("-" * 45)
    for name, acc in results.items():
        print(f"{name:<25} | {acc * 100:.2f}%")
    print("-" * 45)

    # ---- Feature ablation -------------------------------------------------
    print("\n" + "=" * 55)
    print("   FEATURE ABLATION (Logistic Regression, scaled)")
    print("=" * 55)
    print(f"{'Feature set':<28} | {'Test Accuracy':<15}")
    print("-" * 45)
    for label, n_cols in ABLATION_STEPS:
        acc = scaled_logreg_accuracy(
            F_train[:, :n_cols], y_train, F_test[:, :n_cols], y_test
        )
        print(f"{label + f' ({n_cols} features)':<28} | {acc * 100:.2f}%")
    print("-" * 45)

    # ---- Per-class report and confusion matrix (Gradient Boosting) --------
    y_pred_gb = predictions["Gradient Boosting"]
    print("\n--- Classification Report (Gradient Boosting) ---")
    print(classification_report(y_test, y_pred_gb, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred_gb, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.title("Gradient Boosting: Normalized Confusion Matrix (40 features)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
