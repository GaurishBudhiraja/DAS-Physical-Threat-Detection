#!/usr/bin/env python3

"""
H8 — Feature Selection + XGBoost Refinement

Hypothesis:
    Removing redundant engineered features improves generalization
    and reduces computational complexity without sacrificing
    physical-threat detection performance.

Method:
    1. Load frozen H1-H4 representation (2800 features).
    2. Perform day-wise leave-one-day-out evaluation.
    3. For each outer fold:
       - Train an XGBoost importance model using ONLY the 9 training days.
       - Rank features using XGBoost gain importance.
       - Select Top-K features.
       - Train a fresh XGBoost classifier using selected features.
       - Evaluate on the held-out day.
    4. Evaluate Top-500, Top-1000 and Top-1500 representations.

IMPORTANT:
    Feature selection is performed independently inside every outer fold.
    The held-out test day is never used for feature ranking.

Default behavior:
    Run a fast sanity check.

To run the complete experiment:
    RUN_FULL_EXPERIMENT=true python scripts/run_h8_feature_selection.py
"""

import os
import json
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

CACHE_DIR = os.path.join(
    REPO_ROOT,
    "research",
    "cache",
)

FEATURE_PATH = os.path.join(
    CACHE_DIR,
    "h1_h4_features.npy",
)

TARGET_PATH = os.path.join(
    CACHE_DIR,
    "h1_h4_targets.npy",
)

DATETIME_PATH = os.path.join(
    CACHE_DIR,
    "h1_h4_datetimes.npy",
)

RESULTS_DIR = os.path.join(
    REPO_ROOT,
    "results",
    "DAS-XGBoost-H8",
)

RESEARCH_DIR = os.path.join(
    REPO_ROOT,
    "research",
)

EXPECTED_FEATURES = 2800

TOP_K_VALUES = [
    500,
    1000,
    1500,
]

RANDOM_STATE = 42

RUN_FULL_EXPERIMENT = (
    os.environ.get("RUN_FULL_EXPERIMENT", "false").lower()
    == "true"
)


# ============================================================
# Utility functions
# ============================================================

def load_frozen_representation():
    """Load the frozen H1-H4 representation."""

    print("=" * 70)
    print("Loading frozen H1-H4 representation")
    print("=" * 70)

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(
            f"Feature cache not found:\n{FEATURE_PATH}"
        )

    if not os.path.exists(TARGET_PATH):
        raise FileNotFoundError(
            f"Target cache not found:\n{TARGET_PATH}"
        )

    if not os.path.exists(DATETIME_PATH):
        raise FileNotFoundError(
            f"Datetime cache not found:\n{DATETIME_PATH}"
        )

    X = np.load(
        FEATURE_PATH,
        mmap_mode="r",
    )

    y = np.load(
        TARGET_PATH,
    )

    datetimes = np.load(
        DATETIME_PATH,
        allow_pickle=True,
    )

    print(f"Features:   {X.shape}")
    print(f"Targets:    {y.shape}")
    print(f"Datetimes:  {datetimes.shape}")

    return X, y, datetimes


def validate_representation(X, y, datetimes):
    """Validate dimensions, alignment and numerical values."""

    print("\n" + "=" * 70)
    print("Representation validation")
    print("=" * 70)

    assert X.ndim == 2, (
        f"Expected 2D feature matrix, got {X.ndim}D"
    )

    assert X.shape[1] == EXPECTED_FEATURES, (
        f"Expected {EXPECTED_FEATURES} features, "
        f"got {X.shape[1]}"
    )

    assert len(X) == len(y), (
        "Feature/target length mismatch"
    )

    assert len(X) == len(datetimes), (
        "Feature/datetime length mismatch"
    )

    # Check a representative portion first.
    sample_rows = min(5000, len(X))

    sample = np.asarray(
        X[:sample_rows],
        dtype=np.float32,
    )

    assert np.isfinite(sample).all(), (
        "Non-finite values detected in feature cache"
    )

    assert np.isfinite(y).all(), (
        "Non-finite target values detected"
    )

    unique_targets = np.unique(y)

    assert set(unique_targets).issubset({0, 1}), (
        f"Unexpected target classes: {unique_targets}"
    )

    unique_days = np.unique(
        pd.to_datetime(datetimes).strftime("%Y-%m-%d")
    )

    print(f"Feature dimensionality: {X.shape[1]}")
    print(f"Samples:                {len(X)}")
    print(f"Classes:                {unique_targets}")
    print(f"Unique days:            {len(unique_days)}")

    print("\nRepresentation validation: PASS")

    return unique_days


def build_xgb_classifier():
    """Create the XGBoost classifier used by H8."""

    return XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        learning_rate=0.05,
        max_depth=10,
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=8,
        tree_method="hist",
    )


def get_day_strings(datetimes):
    """Convert timestamps to YYYY-MM-DD strings."""

    dt = pd.to_datetime(datetimes)

    return np.asarray(
        dt.strftime("%Y-%m-%d")
    )


def calculate_metrics(y_true, probabilities):
    """Calculate binary physical-threat metrics."""

    # Class 0 = physical proximity / threat.
    threat_probability = 1.0 - probabilities

    predictions = (
        threat_probability >= 0.5
    ).astype(np.int8)

    # predictions above are threat=1.
    # Convert ground truth from:
    #   original class 0 = threat
    #   original class 1 = non-threat
    #
    # Therefore:
    #   threat ground truth = 1 - original y

    y_threat = (
        1 - np.asarray(y_true)
    ).astype(np.int8)

    accuracy = accuracy_score(
        y_threat,
        predictions,
    )

    precision = precision_score(
        y_threat,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_threat,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_threat,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_threat,
        threat_probability,
    )

    cm = confusion_matrix(
        y_threat,
        predictions,
        labels=[1, 0],
    )

    # Rows:
    #   [actual threat, actual non-threat]
    #
    # Columns:
    #   [predicted threat, predicted non-threat]

    missed_threats = int(cm[0, 1])

    false_alerts = int(cm[1, 0])

    return {
        "accuracy": float(accuracy),
        "threat_precision": float(precision),
        "threat_recall": float(recall),
        "threat_f1": float(f1),
        "auc": float(auc),
        "false_proximity_alerts": false_alerts,
        "missed_threats": missed_threats,
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# Feature ranking
# ============================================================

def rank_features(X_train, y_train):
    """
    Train an importance model on training data only
    and return features ranked by XGBoost gain importance.
    """

    print("\nTraining H8 feature-importance model...")

    model = build_xgb_classifier()

    model.fit(
        X_train,
        y_train,
    )

    importance = np.asarray(
        model.feature_importances_,
        dtype=np.float32,
    )

    if importance.shape != (EXPECTED_FEATURES,):
        raise RuntimeError(
            "Unexpected feature importance shape: "
            f"{importance.shape}"
        )

    if not np.isfinite(importance).all():
        raise RuntimeError(
            "Non-finite feature importance detected"
        )

    ranking = np.argsort(
        importance
    )[::-1]

    return model, importance, ranking


def save_feature_ranking(
    importance,
    ranking,
    fold_name,
):
    """Save fold-specific feature importance."""

    os.makedirs(
        RESEARCH_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        RESEARCH_DIR,
        f"h8_feature_importance_{fold_name}.csv",
    )

    dataframe = pd.DataFrame(
        {
            "feature_index": ranking,
            "gain_importance": importance[ranking],
            "rank": np.arange(
                1,
                len(ranking) + 1,
            ),
        }
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# Sanity check
# ============================================================

def run_sanity_check(X, y, datetimes):
    """Run a lightweight H8 validation."""

    print("\n" + "=" * 70)
    print("H8 FEATURE SELECTION SANITY CHECK")
    print("=" * 70)

    unique_days = get_day_strings(
        datetimes
    )

    days = np.unique(
        unique_days
    )

    if len(days) < 2:
        raise RuntimeError(
            "Need at least two unique days."
        )

    # Use the first day as a synthetic held-out day.
    test_day = days[0]

    train_mask = (
        unique_days != test_day
    )

    test_mask = (
        unique_days == test_day
    )

    train_indices = np.flatnonzero(
        train_mask
    )

    test_indices = np.flatnonzero(
        test_mask
    )

    # Use a small subset for the sanity model
    # to avoid accidentally starting the full experiment.
    max_train = min(
        10000,
        len(train_indices),
    )

    max_test = min(
        2000,
        len(test_indices),
    )

    train_indices = train_indices[
        :max_train
    ]

    test_indices = test_indices[
        :max_test
    ]

    X_train = np.asarray(
        X[train_indices],
        dtype=np.float32,
    )

    y_train = np.asarray(
        y[train_indices],
        dtype=np.int8,
    )

    X_test = np.asarray(
        X[test_indices],
        dtype=np.float32,
    )

    y_test = np.asarray(
        y[test_indices],
        dtype=np.int8,
    )

    print(f"\nSynthetic test day: {test_day}")

    print(
        f"Training subset: {X_train.shape}"
    )

    print(
        f"Test subset:     {X_test.shape}"
    )

    # --------------------------------------------------------
    # Feature ranking
    # --------------------------------------------------------

    importance_model, importance, ranking = (
        rank_features(
            X_train,
            y_train,
        )
    )

    print(
        "\nFeature importance shape:",
        importance.shape,
    )

    assert importance.shape == (
        EXPECTED_FEATURES,
    )

    print(
        "Feature importance dimensionality: PASS"
    )

    # --------------------------------------------------------
    # Check Top-K selections
    # --------------------------------------------------------

    for k in TOP_K_VALUES:

        selected = ranking[:k]

        assert len(selected) == k

        assert len(
            np.unique(selected)
        ) == k

        assert selected.min() >= 0

        assert selected.max() < EXPECTED_FEATURES

        X_train_selected = (
            X_train[:, selected]
        )

        X_test_selected = (
            X_test[:, selected]
        )

        assert X_train_selected.shape == (
            len(X_train),
            k,
        )

        assert X_test_selected.shape == (
            len(X_test),
            k,
        )

        assert np.isfinite(
            X_train_selected
        ).all()

        assert np.isfinite(
            X_test_selected
        ).all()

        print(
            f"Top-{k:4d} selection: PASS "
            f"→ {X_train_selected.shape[1]} features"
        )

    # --------------------------------------------------------
    # Verify no test-day information was used
    # --------------------------------------------------------

    assert test_day not in set(
        unique_days[train_indices]
    )

    assert test_day in set(
        unique_days[test_indices]
    )

    print(
        "\nOuter test-day isolation: PASS"
    )

    # --------------------------------------------------------
    # Check actual classifier training on Top-500
    # --------------------------------------------------------

    selected = ranking[:500]

    model = build_xgb_classifier()

    model.fit(
        X_train[:, selected],
        y_train,
    )

    probabilities = model.predict_proba(
        X_test[:, selected]
    )[:, 1]

    assert probabilities.shape == (
        len(X_test),
    )

    assert np.isfinite(
        probabilities
    ).all()

    print(
        "Top-500 classifier training: PASS"
    )

    print(
        "\nH8 SANITY CHECK: PASS"
    )

    print("=" * 70)


# ============================================================
# Full 10-fold experiment
# ============================================================

def run_full_experiment(X, y, datetimes):
    """Run complete day-wise H8 evaluation."""

    print("\n" + "=" * 70)
    print("H8 FULL 10-FOLD EXPERIMENT")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    day_strings = get_day_strings(
        datetimes
    )

    unique_days = np.unique(
        day_strings
    )

    print(
        f"Number of evaluation days: {len(unique_days)}"
    )

    if len(unique_days) != 10:
        print(
            "WARNING: Expected 10 evaluation days."
        )

    all_results = []

    for fold_number, test_day in enumerate(
        unique_days,
        start=1,
    ):

        fold_start = time.time()

        print("\n" + "=" * 70)
        print(
            f"FOLD {fold_number}/{len(unique_days)}"
        )
        print(
            f"Test day: {test_day}"
        )
        print("=" * 70)

        train_mask = (
            day_strings != test_day
        )

        test_mask = (
            day_strings == test_day
        )

        train_indices = np.flatnonzero(
            train_mask
        )

        test_indices = np.flatnonzero(
            test_mask
        )

        print(
            f"Training samples: {len(train_indices)}"
        )

        print(
            f"Test samples:     {len(test_indices)}"
        )

        X_train = np.asarray(
            X[train_indices],
            dtype=np.float32,
        )

        y_train = np.asarray(
            y[train_indices],
            dtype=np.int8,
        )

        X_test = np.asarray(
            X[test_indices],
            dtype=np.float32,
        )

        y_test = np.asarray(
            y[test_indices],
            dtype=np.int8,
        )

        # ----------------------------------------------------
        # Importance model
        # ----------------------------------------------------

        importance_model, importance, ranking = (
            rank_features(
                X_train,
                y_train,
            )
        )

        ranking_path = save_feature_ranking(
            importance,
            ranking,
            f"fold_{fold_number:02d}_{test_day}",
        )

        print(
            f"Saved feature ranking: {ranking_path}"
        )

        # ----------------------------------------------------
        # Evaluate each Top-K
        # ----------------------------------------------------

        for k in TOP_K_VALUES:

            print(
                f"\nTraining H8 Top-{k} model..."
            )

            selected_features = ranking[:k]

            X_train_selected = (
                X_train[:, selected_features]
            )

            X_test_selected = (
                X_test[:, selected_features]
            )

            model = build_xgb_classifier()

            model.fit(
                X_train_selected,
                y_train,
            )

            probabilities = model.predict_proba(
                X_test_selected
            )[:, 1]

            metrics = calculate_metrics(
                y_test,
                probabilities,
            )

            result = {
                "fold": fold_number,
                "test_day": test_day,
                "representation": f"top_{k}",
                "n_features": k,
                "accuracy": metrics["accuracy"],
                "threat_precision": metrics[
                    "threat_precision"
                ],
                "threat_recall": metrics[
                    "threat_recall"
                ],
                "threat_f1": metrics[
                    "threat_f1"
                ],
                "auc": metrics["auc"],
                "false_proximity_alerts": metrics[
                    "false_proximity_alerts"
                ],
                "missed_threats": metrics[
                    "missed_threats"
                ],
                "confusion_matrix": json.dumps(
                    metrics["confusion_matrix"]
                ),
                "fold_runtime_seconds": (
                    time.time() - fold_start
                ),
            }

            all_results.append(
                result
            )

            print(
                f"Top-{k}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Threat F1={metrics['threat_f1']:.4f}, "
                f"AUC={metrics['auc']:.4f}, "
                f"Threat Recall={metrics['threat_recall']:.4f}"
            )

        # ----------------------------------------------------
        # Explicit memory cleanup
        # ----------------------------------------------------

        del X_train
        del X_test
        del y_train
        del y_test

        del importance_model
        del importance
        del ranking

        print(
            f"\nFold runtime: "
            f"{time.time() - fold_start:.1f} seconds"
        )

    # --------------------------------------------------------
    # Save fold metrics
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    fold_metrics_path = os.path.join(
        RESULTS_DIR,
        "h8_fold_metrics.csv",
    )

    results_df.to_csv(
        fold_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = []

    for representation, group in (
        results_df.groupby(
            "representation"
        )
    ):

        summary_rows.append(
            {
                "representation": representation,
                "n_features": int(
                    group["n_features"].iloc[0]
                ),
                "mean_accuracy": group[
                    "accuracy"
                ].mean(),
                "mean_threat_precision": group[
                    "threat_precision"
                ].mean(),
                "mean_threat_recall": group[
                    "threat_recall"
                ].mean(),
                "mean_threat_f1": group[
                    "threat_f1"
                ].mean(),
                "mean_auc": group[
                    "auc"
                ].mean(),
                "total_false_proximity_alerts": int(
                    group[
                        "false_proximity_alerts"
                    ].sum()
                ),
                "total_missed_threats": int(
                    group[
                        "missed_threats"
                    ].sum()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "h8_summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("H8 EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nSummary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nFold metrics saved to:\n"
        f"{fold_metrics_path}"
    )

    print(
        f"Summary saved to:\n"
        f"{summary_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\nH8 — Feature Selection + XGBoost Refinement")

    X, y, datetimes = (
        load_frozen_representation()
    )

    validate_representation(
        X,
        y,
        datetimes,
    )

    if not RUN_FULL_EXPERIMENT:

        print(
            "\nFull experiment is DISABLED."
        )

        print(
            "Running sanity check only."
        )

        run_sanity_check(
            X,
            y,
            datetimes,
        )

        print(
            "\nTo launch the full 10-fold experiment:"
        )

        print(
            "RUN_FULL_EXPERIMENT=true "
            "python scripts/run_h8_feature_selection.py"
        )

        return

    run_full_experiment(
        X,
        y,
        datetimes,
    )


if __name__ == "__main__":
    main()
