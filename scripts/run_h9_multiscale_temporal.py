import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

CACHE_DIR = os.path.join(
    REPO_ROOT,
    "research",
    "cache"
)

FEATURE_PATH = os.path.join(
    CACHE_DIR,
    "h1_h4_features.npy"
)

TARGET_PATH = os.path.join(
    CACHE_DIR,
    "h1_h4_targets.npy"
)

DATETIME_PATH = os.path.join(
    CACHE_DIR,
    "h1_h4_datetimes.npy"
)

RESULTS_DIR = os.path.join(
    REPO_ROOT,
    "results",
    "DAS-XGBoost-H9"
)

EXPECTED_BASE_FEATURES = 2800
EXPECTED_H9_FEATURES = 2900

N_FRAMES = 5

H1_PER_FRAME = 500
H2_PER_FRAME = 5
H3_PER_FRAME = 25

FRAME_FEATURES = (
    H1_PER_FRAME
    + H2_PER_FRAME
    + H3_PER_FRAME
)

H3_TOTAL = H3_PER_FRAME * N_FRAMES

H4_FEATURES = 150
H9_ADDITIONAL = 100


def load_frozen_representation():
    print("=" * 70)
    print("Loading frozen H1-H4 representation")
    print("=" * 70)

    X = np.load(FEATURE_PATH, mmap_mode="r")
    y = np.load(TARGET_PATH)
    datetimes = np.load(DATETIME_PATH, allow_pickle=True)

    print(f"Features:   {X.shape}")
    print(f"Targets:    {y.shape}")
    print(f"Datetimes:  {datetimes.shape}")
    print()

    return X, y, datetimes


def validate_representation(X, y, datetimes):
    print("=" * 70)
    print("Representation validation")
    print("=" * 70)

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix, got {X.ndim}D."
        )

    if X.shape[1] != EXPECTED_BASE_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_BASE_FEATURES} features, "
            f"got {X.shape[1]}."
        )

    if len(X) != len(y) or len(X) != len(datetimes):
        raise ValueError(
            "Feature/target/datetime lengths are misaligned."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Frozen representation contains non-finite values."
        )

    unique_days = np.unique(
        pd.to_datetime(datetimes).strftime("%Y-%m-%d")
    )

    print(f"Feature dimensionality: {X.shape[1]}")
    print(f"Samples:                {len(X)}")
    print(f"Classes:                {np.unique(y)}")
    print(f"Unique days:            {len(unique_days)}")
    print()

    if len(unique_days) != 10:
        raise ValueError(
            f"Expected 10 evaluation days, got {len(unique_days)}."
        )

    print("Representation validation: PASS")
    print()

    return unique_days


def extract_h3_sequence(X):
    """
    Extract the 25 H3 descriptors from each of the five frames.

    Each frame contains:
        H1 = 500
        H2 = 5
        H3 = 25
        ----------------
        530 features

    Therefore H3 occupies the final 25 positions of every
    530-feature frame block.

    Returns:
        h3_sequence: (N, 5, 25)
    """

    n_samples = X.shape[0]

    frame_matrix = X[:, :H3_TOTAL + H1_PER_FRAME * N_FRAMES + H2_PER_FRAME * N_FRAMES]

    expected_frame_features = FRAME_FEATURES * N_FRAMES

    if frame_matrix.shape[1] != expected_frame_features:
        raise ValueError(
            f"Expected first {expected_frame_features} features "
            f"for H1-H3 frame representation, "
            f"got {frame_matrix.shape[1]}."
        )

    frame_matrix = frame_matrix.reshape(
        n_samples,
        N_FRAMES,
        FRAME_FEATURES
    )

    h3_sequence = frame_matrix[:, :, -H3_PER_FRAME:]

    if h3_sequence.shape != (
        n_samples,
        N_FRAMES,
        H3_PER_FRAME
    ):
        raise ValueError(
            f"Unexpected H3 shape: {h3_sequence.shape}"
        )

    return h3_sequence


def build_h9_features(X):
    """
    Construct H9 multi-scale temporal features.

    For each of 25 H3 features:

        First differences:
            mean absolute change
            maximum absolute change

        Second differences:
            mean absolute change
            maximum absolute change

    25 * 4 = 100 H9 features.

    The frozen H1-H4 representation is retained and the
    100 H9 features are appended.
    """

    h3_sequence = extract_h3_sequence(X)

    # ----------------------------------------------------------
    # First-order temporal differences
    # Shape:
    # (N, 4, 25)
    # ----------------------------------------------------------

    first_diff = np.diff(
        h3_sequence,
        axis=1
    )

    mean_abs_first = np.mean(
        np.abs(first_diff),
        axis=1
    )

    max_abs_first = np.max(
        np.abs(first_diff),
        axis=1
    )

    # ----------------------------------------------------------
    # Second-order temporal differences
    # Shape:
    # (N, 3, 25)
    # ----------------------------------------------------------

    second_diff = np.diff(
        first_diff,
        axis=1
    )

    mean_abs_second = np.mean(
        np.abs(second_diff),
        axis=1
    )

    max_abs_second = np.max(
        np.abs(second_diff),
        axis=1
    )

    h9_features = np.concatenate(
        [
            mean_abs_first,
            max_abs_first,
            mean_abs_second,
            max_abs_second,
        ],
        axis=1
    )

    if h9_features.shape[1] != H9_ADDITIONAL:
        raise ValueError(
            f"Expected {H9_ADDITIONAL} H9 features, "
            f"got {h9_features.shape[1]}."
        )

    combined = np.concatenate(
        [
            np.asarray(X, dtype=np.float32),
            h9_features.astype(np.float32),
        ],
        axis=1
    )

    return combined, h3_sequence, h9_features


def run_sanity_check(X, y, datetimes):
    print("=" * 70)
    print("H9 MULTI-SCALE TEMPORAL SANITY CHECK")
    print("=" * 70)

    print("Building H9 representation...")

    # Use a small subset so sanity checking remains memory-safe.
    n_check = min(100, len(X))

    X_small = X[:n_check]

    X_h9, h3_sequence, h9_features = build_h9_features(
        X_small
    )

    print()
    print(f"Input representation:  {X_small.shape}")
    print(f"H3 sequence shape:      {h3_sequence.shape}")
    print(f"H9 additional features: {h9_features.shape}")
    print(f"H9 representation:      {X_h9.shape}")
    print()

    expected_shape = (
        n_check,
        EXPECTED_H9_FEATURES
    )

    if X_h9.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, "
            f"got {X_h9.shape}"
        )

    print("H9 dimensionality: PASS")

    if not np.all(np.isfinite(h9_features)):
        raise ValueError(
            "H9 features contain non-finite values."
        )

    print("Finite-value check: PASS")

    if not np.all(np.isfinite(X_h9)):
        raise ValueError(
            "Combined H9 representation contains non-finite values."
        )

    print("Combined representation finite-value check: PASS")

    # Verify that the original frozen representation is unchanged.
    if not np.array_equal(
        X_h9[:, :EXPECTED_BASE_FEATURES],
        np.asarray(X_small)
    ):
        raise ValueError(
            "Original H1-H4 representation was modified."
        )

    print("Frozen H1-H4 preservation: PASS")

    # Verify expected temporal dimensionality.
    if h3_sequence.shape[1:] != (
        N_FRAMES,
        H3_PER_FRAME
    ):
        raise ValueError(
            "Unexpected temporal H3 structure."
        )

    print("H3 temporal structure: PASS")

    print()
    print("H9 feature composition:")
    print(f"  H1-H4 representation: {EXPECTED_BASE_FEATURES}")
    print(f"  H9 temporal features:  {H9_ADDITIONAL}")
    print(f"  Total:                 {EXPECTED_H9_FEATURES}")
    print()

    print("H9 temporal descriptors:")
    print("  1. Mean absolute first difference")
    print("  2. Maximum absolute first difference")
    print("  3. Mean absolute second difference")
    print("  4. Maximum absolute second difference")
    print()

    print("Feature statistics:")
    print(f"  Min:  {np.min(h9_features):.6f}")
    print(f"  Max:  {np.max(h9_features):.6f}")
    print(f"  Mean: {np.mean(h9_features):.6f}")
    print(f"  Std:  {np.std(h9_features):.6f}")
    print()

    print("H9 SANITY CHECK: PASS")
    print("=" * 70)
    print()

    print(
        "To launch the full 10-fold experiment:"
    )
    print()
    print(
        "RUN_FULL_EXPERIMENT=true "
        "python scripts/run_h9_multiscale_temporal.py"
    )


def create_classifier():
    return XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        learning_rate=0.05,
        max_depth=10,
        n_estimators=500,
        random_state=42,
        n_jobs=8,
        tree_method="hist",
    )


def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= 0.5
    ).astype(np.int8)

    # Class 0 is the physical-threat class.
    threat_true = (y_true == 0).astype(np.int8)
    threat_pred = (predictions == 0).astype(np.int8)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    threat_precision = precision_score(
        threat_true,
        threat_pred,
        zero_division=0
    )

    threat_recall = recall_score(
        threat_true,
        threat_pred,
        zero_division=0
    )

    threat_f1 = f1_score(
        threat_true,
        threat_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    false_proximity_alerts = int(cm[1, 0])
    missed_threats = int(cm[0, 1])

    return {
        "accuracy": accuracy,
        "threat_precision": threat_precision,
        "threat_recall": threat_recall,
        "threat_f1": threat_f1,
        "auc": auc,
        "false_proximity_alerts": false_proximity_alerts,
        "missed_threats": missed_threats,
        "tn_non_threat": int(cm[1, 1]),
        "tp_threat": int(cm[0, 0]),
    }


def run_full_experiment(X, y, datetimes, unique_days):
    print("=" * 70)
    print("H9 FULL 10-FOLD EXPERIMENT")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print("Building H9 representation...")

    start_build = time.time()

    X_h9, _, _ = build_h9_features(X)

    build_time = time.time() - start_build

    print(
        f"H9 representation: {X_h9.shape}"
    )

    print(
        f"H9 construction time: "
        f"{build_time:.1f} seconds"
    )

    print()

    fold_results = []

    for fold_idx, test_day in enumerate(
        unique_days,
        start=1
    ):
        fold_start = time.time()

        print("=" * 70)
        print(
            f"FOLD {fold_idx}/10"
        )
        print(
            f"Test day: {test_day}"
        )
        print("=" * 70)

        day_strings = pd.to_datetime(
            datetimes
        ).strftime("%Y-%m-%d")

        test_mask = (
            day_strings == str(test_day)
        )

        train_mask = ~test_mask

        X_train = X_h9[train_mask]
        y_train = y[train_mask]

        X_test = X_h9[test_mask]
        y_test = y[test_mask]

        print(
            f"Training samples: {len(X_train)}"
        )

        print(
            f"Test samples:     {len(X_test)}"
        )

        print()

        model = create_classifier()

        print("Training H9 XGBoost model...")

        model.fit(
            X_train,
            y_train
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        metrics = calculate_metrics(
            y_test,
            probabilities
        )

        metrics["fold"] = fold_idx
        metrics["test_day"] = str(test_day)
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)

        fold_results.append(metrics)

        print(
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Threat F1={metrics['threat_f1']:.4f}, "
            f"AUC={metrics['auc']:.4f}, "
            f"Threat Recall={metrics['threat_recall']:.4f}"
        )

        print(
            f"False proximity alerts="
            f"{metrics['false_proximity_alerts']}"
        )

        print(
            f"Missed threats="
            f"{metrics['missed_threats']}"
        )

        print(
            f"Fold runtime: "
            f"{time.time() - fold_start:.1f} seconds"
        )

        print()

    fold_df = pd.DataFrame(
        fold_results
    )

    fold_path = os.path.join(
        RESULTS_DIR,
        "h9_fold_metrics.csv"
    )

    fold_df.to_csv(
        fold_path,
        index=False
    )

    summary = {
        "representation": "H1-H4 + H9",
        "n_features": EXPECTED_H9_FEATURES,
        "mean_accuracy":
            float(fold_df["accuracy"].mean()),
        "mean_threat_precision":
            float(
                fold_df[
                    "threat_precision"
                ].mean()
            ),
        "mean_threat_recall":
            float(
                fold_df[
                    "threat_recall"
                ].mean()
            ),
        "mean_threat_f1":
            float(
                fold_df[
                    "threat_f1"
                ].mean()
            ),
        "mean_auc":
            float(fold_df["auc"].mean()),
        "total_false_proximity_alerts":
            int(
                fold_df[
                    "false_proximity_alerts"
                ].sum()
            ),
        "total_missed_threats":
            int(
                fold_df[
                    "missed_threats"
                ].sum()
            ),
    }

    summary_path = os.path.join(
        RESULTS_DIR,
        "h9_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print("=" * 70)
    print("H9 EXPERIMENT COMPLETE")
    print("=" * 70)
    print()

    print("Summary:")
    print(
        pd.DataFrame([summary]).to_string(
            index=False
        )
    )

    print()
    print(
        f"Fold metrics saved to:\n"
        f"{fold_path}"
    )

    print(
        f"Summary saved to:\n"
        f"{summary_path}"
    )


def main():
    print()
    print(
        "H9 — Multi-scale Temporal Dynamics"
    )
    print("=" * 70)

    X, y, datetimes = (
        load_frozen_representation()
    )

    unique_days = validate_representation(
        X,
        y,
        datetimes
    )

    run_full = (
        os.environ.get(
            "RUN_FULL_EXPERIMENT",
            "false"
        ).lower()
        == "true"
    )

    if not run_full:
        print(
            "Full experiment is DISABLED."
        )
        print(
            "Running sanity check only."
        )
        print()

        run_sanity_check(
            X,
            y,
            datetimes
        )

        return

    run_full_experiment(
        X,
        y,
        datetimes,
        unique_days
    )


if __name__ == "__main__":
    main()
