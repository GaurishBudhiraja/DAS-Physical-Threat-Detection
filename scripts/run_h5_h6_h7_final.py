#!/usr/bin/env python3

import os
import sys
import json
import warnings
from pathlib import Path

import h5py
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = REPO_ROOT / "research" / "cache"

FEATURE_PATH = CACHE_DIR / "h1_h4_features.npy"
TARGET_PATH = CACHE_DIR / "h1_h4_targets.npy"
DATETIME_PATH = CACHE_DIR / "h1_h4_datetimes.npy"

H5_PATH = (
    REPO_ROOT
    / "data"
    / "dataset_sensor_range_1440_1690_0.h5"
)

RESULT_DIR = (
    REPO_ROOT
    / "results"
    / "DAS-XGBoost-H5-H6-H7"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


EXPECTED_FEATURES = 2800

THREAT_CLASS = 0

# H5
AUX_FEATURE_START = 2500
AUX_FEATURE_END = 2800
RIDGE_ALPHA = 100.0

# H6
MIN_THREAT_RECALL = 0.90
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)

# H7
PERSISTENCE_WINDOW = 3
PERSISTENCE_REQUIRED = 2

# Reproducibility
RANDOM_STATE = 42

# XGBoost
XGB_PARAMS = dict(
    objective="binary:logistic",
    booster="gbtree",
    learning_rate=0.05,
    max_depth=10,
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=8,
    tree_method="hist",
)


# ============================================================
# DATA LOADING
# ============================================================

def load_cached_data():
    print("Loading cached H1-H4 representation...")

    X = np.load(
        FEATURE_PATH,
        mmap_mode="r",
    )

    y = np.load(TARGET_PATH)

    dt = np.load(
        DATETIME_PATH,
        allow_pickle=True,
    )

    print(f"Cached features: {X.shape}")
    print(f"Binary targets:  {y.shape}")
    print(f"Datetimes:       {dt.shape}")

    if X.ndim != 2:
        raise ValueError(f"Expected 2-D features, got {X.ndim}-D.")

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {X.shape[1]}."
        )

    if len(X) != len(y) or len(X) != len(dt):
        raise ValueError("Feature/target/datetime lengths do not match.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Targets contain non-finite values.")

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Targets are not binary.")

    print("H5-H7 cache dimensionality/alignment: PASS")

    return X, y.astype(np.int8), dt


def load_raw_distance():
    print("\nLoading raw continuous distance...")

    with h5py.File(H5_PATH, "r") as f:
        raw_distance = np.asarray(
            f["y"],
            dtype=np.float32,
        )

        raw_dt = np.asarray(
            [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["datetimes"]
            ]
        )

    print(f"Raw distance: {raw_distance.shape}")
    print(f"Raw datetimes: {raw_dt.shape}")

    if len(raw_distance) != len(raw_dt):
        raise ValueError("Raw distance/datetime lengths differ.")

    return raw_distance, raw_dt


def align_raw_distance_to_cache(
    raw_distance,
    raw_dt,
    cached_dt,
):
    """
    The cached H1-H4 representation uses the same reducer timestamps
    as the original classification pipeline.

    For each cached timestamp, recover the corresponding raw continuous
    distance. Duplicate timestamps are handled deterministically by
    consuming raw observations in chronological order.
    """

    print("\nAligning continuous distance to cached representation...")

    buckets = {}

    for distance, timestamp in zip(
        raw_distance,
        raw_dt,
    ):
        buckets.setdefault(timestamp, []).append(float(distance))

    counters = {}

    reduced_distance = np.empty(
        len(cached_dt),
        dtype=np.float32,
    )

    missing = 0

    for i, timestamp in enumerate(cached_dt):
        values = buckets.get(timestamp)

        if values is None:
            missing += 1
            continue

        index = counters.get(timestamp, 0)

        if index >= len(values):
            index = len(values) - 1

        reduced_distance[i] = values[index]
        counters[timestamp] = index + 1

    if missing:
        raise ValueError(
            f"Could not align {missing} cached timestamps."
        )

    if not np.all(np.isfinite(reduced_distance)):
        raise ValueError(
            "Aligned distance contains non-finite values."
        )

    print(
        f"Aligned continuous distance: "
        f"{reduced_distance.shape}"
    )

    print("Continuous-distance alignment: PASS")

    return reduced_distance


# ============================================================
# DAY HANDLING
# ============================================================

def normalize_datetime_array(dt):
    return np.asarray(
        [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in dt
        ]
    )


def extract_days(dt):
    dt = normalize_datetime_array(dt)

    return np.asarray(
        [
            timestamp[:10]
            for timestamp in dt
        ]
    )


def get_day_indices(days):
    unique_days = np.unique(days)

    result = {}

    for day in unique_days:
        result[day] = np.where(days == day)[0]

    return result


# ============================================================
# H5 — DISTANCE AUXILIARY MODEL
# ============================================================

def build_ridge():
    return Ridge(
        alpha=RIDGE_ALPHA,
        solver="lsqr",
        max_iter=1000,
    )


def make_auxiliary_features(X):
    """
    H3 + H4 features.

    H1: 2500
    H2: 25
    H3: 125
    H4: 150

    H3 + H4 = 275 features.
    """

    return X[
        :,
        AUX_FEATURE_START:AUX_FEATURE_END,
    ]


def crossfit_distance_for_training(
    X,
    distance,
    days,
    training_days,
):
    """
    Produce strictly cross-fitted distance predictions for all
    outer-training observations.

    The outer test day is never used here.

    Training days are split into two groups:

        Group A -> predict Group B
        Group B -> predict Group A

    This gives leakage-safe distance predictions for the classifier's
    training observations.
    """

    print("\nH5 distance cross-fitting...")

    training_days = list(training_days)

    midpoint = len(training_days) // 2

    group_a_days = training_days[:midpoint]
    group_b_days = training_days[midpoint:]

    group_a = np.concatenate(
        [np.where(days == d)[0] for d in group_a_days]
    )

    group_b = np.concatenate(
        [np.where(days == d)[0] for d in group_b_days]
    )

    print(
        f"  Group A: {len(group_a_days)} days, "
        f"{len(group_a)} samples"
    )

    print(
        f"  Group B: {len(group_b_days)} days, "
        f"{len(group_b)} samples"
    )

    X_aux = make_auxiliary_features(X)

    predictions = np.empty(
        len(training_days) and len(group_a) + len(group_b),
        dtype=np.float32,
    )

    # The indices in the prediction array must correspond to global
    # dataset indices, so allocate by full dataset length.
    predictions = np.empty(
        len(X),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # A -> B
    # --------------------------------------------------------

    print("  Training Ridge distance model A -> B...")

    model_a = build_ridge()

    model_a.fit(
        np.asarray(X_aux[group_a], dtype=np.float32),
        distance[group_a],
    )

    predictions[group_b] = model_a.predict(
        np.asarray(X_aux[group_b], dtype=np.float32)
    ).astype(np.float32)

    del model_a

    # --------------------------------------------------------
    # B -> A
    # --------------------------------------------------------

    print("  Training Ridge distance model B -> A...")

    model_b = build_ridge()

    model_b.fit(
        np.asarray(X_aux[group_b], dtype=np.float32),
        distance[group_b],
    )

    predictions[group_a] = model_b.predict(
        np.asarray(X_aux[group_a], dtype=np.float32)
    ).astype(np.float32)

    del model_b

    print("  Training cross-fitted distance predictions: PASS")

    return predictions


def train_final_distance_model(
    X,
    distance,
    days,
    training_days,
    test_indices,
):
    """
    Train final distance model using all outer-training days and
    predict the completely held-out outer test day.
    """

    training_indices = np.concatenate(
        [np.where(days == d)[0] for d in training_days]
    )

    X_aux = make_auxiliary_features(X)

    print(
        "  Training final Ridge distance model "
        "on all outer-training days..."
    )

    model = build_ridge()

    model.fit(
        np.asarray(X_aux[training_indices], dtype=np.float32),
        distance[training_indices],
    )

    test_prediction = model.predict(
        np.asarray(X_aux[test_indices], dtype=np.float32)
    ).astype(np.float32)

    del model

    return test_prediction


# ============================================================
# H6 — CALIBRATION
# ============================================================

def class1_probability_to_threat_probability(
    class1_probability,
):
    """
    Class 0 is physical proximity / threat.
    Class 1 is non-proximity.

    XGBoost returns P(class=1), therefore:

        P(threat) = 1 - P(class=1)
    """

    return 1.0 - class1_probability


def fit_calibrator(
    raw_threat_probability,
    y_validation,
):
    threat_target = (
        y_validation == THREAT_CLASS
    ).astype(np.int8)

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        raw_threat_probability,
        threat_target,
    )

    return calibrator


def select_threat_threshold(
    threat_probability,
    y_true,
):
    threat_true = (
        y_true == THREAT_CLASS
    ).astype(np.int8)

    best = None

    for threshold in THRESHOLD_GRID:

        prediction = (
            threat_probability >= threshold
        ).astype(np.int8)

        recall = recall_score(
            threat_true,
            prediction,
            zero_division=0,
        )

        false_alerts = int(
            np.sum(
                (y_true != THREAT_CLASS)
                & (prediction == 1)
            )
        )

        if recall < MIN_THREAT_RECALL:
            continue

        candidate = (
            false_alerts,
            -recall,
            threshold,
        )

        if best is None or candidate < best[0]:
            best = (
                candidate,
                threshold,
                recall,
                false_alerts,
            )

    if best is None:
        # Fall back to the lowest threshold.
        threshold = float(THRESHOLD_GRID[0])

        prediction = (
            threat_probability >= threshold
        ).astype(np.int8)

        recall = recall_score(
            threat_true,
            prediction,
            zero_division=0,
        )

        false_alerts = int(
            np.sum(
                (y_true != THREAT_CLASS)
                & (prediction == 1)
            )
        )

        return threshold, recall, false_alerts

    return (
        best[1],
        best[2],
        best[3],
    )


# ============================================================
# H7 — TEMPORAL PERSISTENCE
# ============================================================

def apply_temporal_persistence(
    threat_decisions,
    dt,
    window=PERSISTENCE_WINDOW,
    required=PERSISTENCE_REQUIRED,
):
    """
    Apply persistence independently within each calendar day.

    No state is carried from one day into another.
    """

    decisions = np.asarray(
        threat_decisions,
        dtype=bool,
    )

    dt = normalize_datetime_array(dt)

    days = extract_days(dt)

    output = np.zeros(
        len(decisions),
        dtype=bool,
    )

    for day in np.unique(days):

        indices = np.where(
            days == day
        )[0]

        day_decisions = decisions[indices]

        for local_i in range(len(indices)):

            start = max(
                0,
                local_i - window + 1,
            )

            recent = day_decisions[
                start:local_i + 1
            ]

            if np.sum(recent) >= required:
                output[indices[local_i]] = True

    return output


# ============================================================
# CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    y_train,
):
    model = XGBClassifier(
        **XGB_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    threat_prediction,
    raw_threat_probability,
    calibrated_threat_probability,
    threshold,
):
    threat_true = (
        y_true == THREAT_CLASS
    ).astype(np.int8)

    threat_prediction = np.asarray(
        threat_prediction,
        dtype=np.int8,
    )

    cm = confusion_matrix(
        threat_true,
        threat_prediction,
        labels=[1, 0],
    )

    # labels=[1,0]:
    #
    # [[TP, FN],
    #  [FP, TN]]

    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    return {
        "accuracy": float(
            accuracy_score(
                threat_true,
                threat_prediction,
            )
        ),
        "threat_precision": float(
            precision_score(
                threat_true,
                threat_prediction,
                zero_division=0,
            )
        ),
        "threat_recall": float(
            recall_score(
                threat_true,
                threat_prediction,
                zero_division=0,
            )
        ),
        "threat_f1": float(
            f1_score(
                threat_true,
                threat_prediction,
                zero_division=0,
            )
        ),
        "roc_auc_raw": float(
            roc_auc_score(
                threat_true,
                raw_threat_probability,
            )
        ),
        "brier_raw": float(
            brier_score_loss(
                threat_true,
                raw_threat_probability,
            )
        ),
        "brier_calibrated": float(
            brier_score_loss(
                threat_true,
                calibrated_threat_probability,
            )
        ),
        "threshold": float(threshold),
        "true_threats": int(tp + fn),
        "correct_threats": int(tp),
        "missed_threats": int(fn),
        "false_proximity_alerts": int(fp),
        "correct_non_threats": int(tn),
        "total": int(len(y_true)),
        "confusion_tp": tp,
        "confusion_fn": fn,
        "confusion_fp": fp,
        "confusion_tn": tn,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    print("=" * 70)
    print("REAL H5 + H6 + H7 EXPERIMENT")
    print("=" * 70)

    X, y, dt = load_cached_data()

    raw_distance, raw_dt = load_raw_distance()

    distance = align_raw_distance_to_cache(
        raw_distance,
        raw_dt,
        dt,
    )

    days = extract_days(dt)

    day_indices = get_day_indices(days)

    unique_days = list(
        np.sort(
            np.unique(days)
        )
    )

    print("\nUnique days:")

    for day in unique_days:
        print(
            f"  {day}: "
            f"{len(day_indices[day])} samples"
        )

    fold_results = []

    # ========================================================
    # OUTER DAY-WISE LEAVE-ONE-DAY-OUT
    # ========================================================

    for fold_number, test_day in enumerate(
        unique_days,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(
            f"FOLD {fold_number}/{len(unique_days)} "
            f"— TEST DAY {test_day}"
        )
        print("=" * 70)

        training_days = [
            d for d in unique_days
            if d != test_day
        ]

        test_indices = day_indices[test_day]

        training_indices = np.concatenate(
            [
                day_indices[d]
                for d in training_days
            ]
        )

        # ----------------------------------------------------
        # H5
        # ----------------------------------------------------

        h5_training_distance = (
            crossfit_distance_for_training(
                X,
                distance,
                days,
                training_days,
            )
        )

        h5_test_distance = (
            train_final_distance_model(
                X,
                distance,
                days,
                training_days,
                test_indices,
            )
        )

        X_train_h5 = np.asarray(
            X[training_indices],
            dtype=np.float32,
        )

        X_test_h5 = np.asarray(
            X[test_indices],
            dtype=np.float32,
        )

        X_train_h5 = np.column_stack(
            [
                X_train_h5,
                h5_training_distance[
                    training_indices
                ],
            ]
        ).astype(np.float32)

        X_test_h5 = np.column_stack(
            [
                X_test_h5,
                h5_test_distance,
            ]
        ).astype(np.float32)

        print(
            f"H5 classifier training shape: "
            f"{X_train_h5.shape}"
        )

        print(
            f"H5 classifier test shape: "
            f"{X_test_h5.shape}"
        )

        # ----------------------------------------------------
        # H6 VALIDATION SPLIT
        #
        # Use the final training day as validation day.
        # It remains completely separate from the outer test day.
        # ----------------------------------------------------

        validation_day = training_days[-1]

        classifier_training_days = training_days[:-1]

        classifier_training_indices = np.concatenate(
            [
                day_indices[d]
                for d in classifier_training_days
            ]
        )

        validation_indices = day_indices[
            validation_day
        ]

        print(
            f"\nH6 validation day: "
            f"{validation_day}"
        )

        print(
            "Training validation classifier..."
        )

        X_validation_train = X_train_h5[
            np.isin(
                training_indices,
                classifier_training_indices,
            )
        ]

        y_validation_train = y[
            classifier_training_indices
        ]

        X_validation = X_train_h5[
            np.isin(
                training_indices,
                validation_indices,
            )
        ]

        y_validation = y[
            validation_indices
        ]

        validation_classifier = train_classifier(
            X_validation_train,
            y_validation_train,
        )

        validation_class1_probability = (
            validation_classifier.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_raw_threat_probability = (
            class1_probability_to_threat_probability(
                validation_class1_probability
            )
        )

        calibrator = fit_calibrator(
            validation_raw_threat_probability,
            y_validation,
        )

        validation_calibrated_probability = (
            calibrator.predict(
                validation_raw_threat_probability
            )
        )

        (
            threshold,
            validation_threat_recall,
            validation_false_alerts,
        ) = select_threat_threshold(
            validation_calibrated_probability,
            y_validation,
        )

        print(
            f"H6 selected threshold: "
            f"{threshold:.3f}"
        )

        print(
            f"H6 validation threat recall: "
            f"{validation_threat_recall:.4f}"
        )

        print(
            f"H6 validation false alerts: "
            f"{validation_false_alerts}"
        )

        del validation_classifier

        # ----------------------------------------------------
        # FINAL OUTER CLASSIFIER
        # ----------------------------------------------------

        print(
            "\nTraining final outer classifier..."
        )

        y_train = y[
            training_indices
        ]

        y_test = y[
            test_indices
        ]

        final_classifier = train_classifier(
            X_train_h5,
            y_train,
        )

        class1_probability = (
            final_classifier.predict_proba(
                X_test_h5
            )[:, 1]
        )

        raw_threat_probability = (
            class1_probability_to_threat_probability(
                class1_probability
            )
        )

        calibrated_threat_probability = (
            calibrator.predict(
                raw_threat_probability
            )
        )

        # ----------------------------------------------------
        # H7
        # ----------------------------------------------------

        raw_threat_decision = (
            calibrated_threat_probability
            >= threshold
        )

        persistent_threat_decision = (
            apply_temporal_persistence(
                raw_threat_decision,
                dt[test_indices],
            )
        )

        metrics = calculate_metrics(
            y_test,
            persistent_threat_decision,
            raw_threat_probability,
            calibrated_threat_probability,
            threshold,
        )

        metrics["fold"] = fold_number
        metrics["test_day"] = test_day
        metrics["validation_day"] = validation_day
        metrics["validation_threat_recall"] = (
            float(validation_threat_recall)
        )
        metrics["validation_false_alerts"] = (
            int(validation_false_alerts)
        )

        fold_results.append(metrics)

        print("\nFINAL H5 + H6 + H7 FOLD RESULT")
        print(
            f"Accuracy:              "
            f"{metrics['accuracy']:.4f}"
        )
        print(
            f"Threat precision:      "
            f"{metrics['threat_precision']:.4f}"
        )
        print(
            f"Threat recall:         "
            f"{metrics['threat_recall']:.4f}"
        )
        print(
            f"Threat F1:             "
            f"{metrics['threat_f1']:.4f}"
        )
        print(
            f"ROC-AUC:               "
            f"{metrics['roc_auc_raw']:.4f}"
        )
        print(
            f"Brier raw:             "
            f"{metrics['brier_raw']:.6f}"
        )
        print(
            f"Brier calibrated:      "
            f"{metrics['brier_calibrated']:.6f}"
        )
        print(
            f"False proximity alerts:"
            f" {metrics['false_proximity_alerts']}"
        )
        print(
            f"Missed threats:        "
            f"{metrics['missed_threats']}"
        )

        del final_classifier
        del X_train_h5
        del X_test_h5
        del h5_training_distance
        del h5_test_distance

    # ========================================================
    # FINAL POOLED METRICS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL REAL H5 + H6 + H7 RESULTS")
    print("=" * 70)

    total_tp = sum(
        r["confusion_tp"]
        for r in fold_results
    )

    total_fn = sum(
        r["confusion_fn"]
        for r in fold_results
    )

    total_fp = sum(
        r["confusion_fp"]
        for r in fold_results
    )

    total_tn = sum(
        r["confusion_tn"]
        for r in fold_results
    )

    total = (
        total_tp
        + total_fn
        + total_fp
        + total_tn
    )

    pooled_accuracy = (
        total_tp + total_tn
    ) / total

    pooled_precision = (
        total_tp
        / max(
            total_tp + total_fp,
            1,
        )
    )

    pooled_recall = (
        total_tp
        / max(
            total_tp + total_fn,
            1,
        )
    )

    pooled_f1 = (
        2
        * pooled_precision
        * pooled_recall
        / max(
            pooled_precision + pooled_recall,
            1e-12,
        )
    )

    mean_auc = float(
        np.mean(
            [
                r["roc_auc_raw"]
                for r in fold_results
            ]
        )
    )

    mean_brier_raw = float(
        np.mean(
            [
                r["brier_raw"]
                for r in fold_results
            ]
        )
    )

    mean_brier_calibrated = float(
        np.mean(
            [
                r["brier_calibrated"]
                for r in fold_results
            ]
        )
    )

    summary = {
        "accuracy": pooled_accuracy,
        "threat_precision": pooled_precision,
        "threat_recall": pooled_recall,
        "threat_f1": pooled_f1,
        "mean_fold_roc_auc": mean_auc,
        "mean_fold_brier_raw": mean_brier_raw,
        "mean_fold_brier_calibrated": mean_brier_calibrated,
        "true_threats": total_tp + total_fn,
        "correct_threats": total_tp,
        "missed_threats": total_fn,
        "false_proximity_alerts": total_fp,
        "correct_non_threats": total_tn,
        "total": total,
        "tp": total_tp,
        "fn": total_fn,
        "fp": total_fp,
        "tn": total_tn,
        "persistence_window": PERSISTENCE_WINDOW,
        "persistence_required": PERSISTENCE_REQUIRED,
        "minimum_validation_threat_recall": MIN_THREAT_RECALL,
    }

    print(
        f"\nAccuracy:              "
        f"{pooled_accuracy:.4f}"
    )
    print(
        f"Threat precision:      "
        f"{pooled_precision:.4f}"
    )
    print(
        f"Threat recall:         "
        f"{pooled_recall:.4f}"
    )
    print(
        f"Threat F1:             "
        f"{pooled_f1:.4f}"
    )
    print(
        f"Mean fold ROC-AUC:     "
        f"{mean_auc:.4f}"
    )
    print(
        f"Mean raw Brier:        "
        f"{mean_brier_raw:.6f}"
    )
    print(
        f"Mean calibrated Brier: "
        f"{mean_brier_calibrated:.6f}"
    )
    print(
        f"False proximity alerts:"
        f" {total_fp}"
    )
    print(
        f"Missed threats:        "
        f"{total_fn}"
    )

    print("\nConfusion matrix:")
    print(
        "                Pred Threat  Pred Non-Threat"
    )
    print(
        f"Actual Threat      {total_tp:8d}"
        f"       {total_fn:8d}"
    )
    print(
        f"Actual Non-Threat  {total_fp:8d}"
        f"       {total_tn:8d}"
    )

    # --------------------------------------------------------
    # Save fold CSV
    # --------------------------------------------------------

    fold_csv = RESULT_DIR / "h5_h6_h7_fold_metrics.csv"

    columns = [
        "fold",
        "test_day",
        "validation_day",
        "accuracy",
        "threat_precision",
        "threat_recall",
        "threat_f1",
        "roc_auc_raw",
        "brier_raw",
        "brier_calibrated",
        "threshold",
        "validation_threat_recall",
        "validation_false_alerts",
        "true_threats",
        "correct_threats",
        "missed_threats",
        "false_proximity_alerts",
        "correct_non_threats",
        "confusion_tp",
        "confusion_fn",
        "confusion_fp",
        "confusion_tn",
        "total",
    ]

    with open(
        fold_csv,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            ",".join(columns)
            + "\n"
        )

        for row in fold_results:

            f.write(
                ",".join(
                    str(row[c])
                    for c in columns
                )
                + "\n"
            )

    summary_json = RESULT_DIR / "h5_h6_h7_summary.json"

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nArtifacts saved:")
    print(f"  {fold_csv}")
    print(f"  {summary_json}")

    print("\n" + "=" * 70)
    print("H5 + H6 + H7 REAL EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
