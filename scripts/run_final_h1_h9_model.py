#!/usr/bin/env python3

"""
FINAL H1-H9 DAS PHYSICAL THREAT DETECTION EXPERIMENT

Pipeline:
    H1-H4 frozen representation
        + H9 higher-order temporal dynamics
        -> H8 fold-contained Top-1000 selection
        -> H5 distance-aware auxiliary prediction
        -> XGBoost classifier
        -> H6 isotonic calibration + threat-aware threshold
        -> H7 temporal persistence

Evaluation:
    10-fold leave-one-day-out evaluation.

Important:
    Ground-truth distance is NEVER used as a classifier input.
    Distance is used only as an auxiliary H5 supervision target.

    H6 calibration/threshold selection is performed on an
    inner validation day that is excluded from the final
    outer classifier training.

Outputs:
    results/DAS-XGBoost-FINAL/
"""

import os
import sys
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, REPO_ROOT)

CACHE_DIR = os.path.join(
    REPO_ROOT,
    "research",
    "cache",
)

RESULTS_DIR = os.path.join(
    REPO_ROOT,
    "results",
    "DAS-XGBoost-FINAL",
)

os.makedirs(RESULTS_DIR, exist_ok=True)


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


# ============================================================
# CONFIGURATION
# ============================================================

H1_H4_FEATURES = 2800
H9_ADDITIONAL = 100
BASE_FEATURES = H1_H4_FEATURES + H9_ADDITIONAL

H1_PER_FRAME = 500
H2_PER_FRAME = 5
H3_PER_FRAME = 25

N_FRAMES = 5
FRAME_FEATURES = 530

H3_TOTAL = H3_PER_FRAME * N_FRAMES

H4_FEATURES = 150

# H3 + H4 auxiliary distance features
AUX_FEATURE_START = (
    H1_PER_FRAME * N_FRAMES
    + H2_PER_FRAME * N_FRAMES
)

AUX_FEATURE_END = (
    AUX_FEATURE_START
    + H3_TOTAL
    + H4_FEATURES
)

TOP_K = 1000

THREAT_CLASS = 0

MIN_THREAT_RECALL = 0.90

THRESHOLD_GRID = np.linspace(
    0.01,
    0.99,
    99,
)

RIDGE_ALPHA = 100.0

PERSISTENCE_WINDOW = 3
PERSISTENCE_REQUIRED = 2


XGB_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "learning_rate": 0.05,
    "max_depth": 10,
    "n_estimators": 500,
    "random_state": 42,
    "n_jobs": 8,
    "tree_method": "hist",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_cached_data():

    print("Loading frozen H1-H4 representation...")

    X = np.load(
        FEATURE_PATH,
        mmap_mode="r",
    )

    y = np.load(
        TARGET_PATH,
    )

    dt = np.load(
        DATETIME_PATH,
        allow_pickle=True,
    )

    print(f"Features:   {X.shape}")
    print(f"Targets:    {y.shape}")
    print(f"Datetimes:  {dt.shape}")

    if X.shape[1] != H1_H4_FEATURES:
        raise ValueError(
            f"Expected {H1_H4_FEATURES} features, "
            f"got {X.shape[1]}"
        )

    if len(X) != len(y) or len(X) != len(dt):
        raise ValueError(
            "Feature/target/datetime alignment failure."
        )

    print("Frozen cache validation: PASS")

    return X, y.astype(np.int8), dt


def load_raw_distance():

    print("\nLoading raw continuous distance...")

    import h5py

    h5_path = os.path.join(
        REPO_ROOT,
        "data",
        "dataset_sensor_range_1440_1690_0.h5",
    )

    with h5py.File(h5_path, "r") as f:

        distance = np.array(
            f["y"],
            dtype=np.float32,
        )

        raw_dt = np.array(
            [
                x.decode("utf-8")
                if isinstance(x, bytes)
                else str(x)
                for x in f["datetimes"]
            ]
        )

    print(f"Raw distance: {distance.shape}")
    print(f"Raw datetime: {raw_dt.shape}")

    return distance, raw_dt


# ============================================================
# DATETIME HELPERS
# ============================================================

def normalize_datetime_array(dt):

    result = []

    for value in dt:

        if isinstance(value, np.datetime64):
            value = str(value)

        result.append(
            pd.Timestamp(str(value))
        )

    return np.asarray(
        result,
        dtype=object,
    )


def extract_days(dt):

    normalized = normalize_datetime_array(dt)

    return np.asarray(
        [
            timestamp.strftime("%Y-%m-%d")
            for timestamp in normalized
        ],
        dtype=str,
    )


# ============================================================
# DISTANCE ALIGNMENT
# ============================================================

def align_raw_distance_to_cache(
    raw_distance,
    raw_dt,
    cached_dt,
):

    raw_distance = np.asarray(
        raw_distance,
        dtype=np.float32,
    )

    raw_dt_normalized = normalize_datetime_array(
        raw_dt
    )

    cached_dt_normalized = normalize_datetime_array(
        cached_dt
    )

    raw_map = {
        timestamp: distance
        for timestamp, distance
        in zip(
            raw_dt_normalized,
            raw_distance,
        )
    }

    aligned = np.empty(
        len(cached_dt_normalized),
        dtype=np.float32,
    )

    missing = 0

    for i, timestamp in enumerate(
        cached_dt_normalized
    ):

        if timestamp not in raw_map:
            missing += 1
            continue

        aligned[i] = raw_map[timestamp]

    if missing:
        raise ValueError(
            f"Could not align {missing} cached timestamps."
        )

    if not np.isfinite(aligned).all():
        raise ValueError(
            "Aligned distance contains non-finite values."
        )

    print(
        f"Aligned continuous distance: "
        f"{aligned.shape}"
    )

    print(
        "Continuous-distance alignment: PASS"
    )

    return aligned


# ============================================================
# H9
# ============================================================

def extract_h3_sequence(X):

    n_samples = X.shape[0]

    frame_matrix = X[
        :,
        :FRAME_FEATURES * N_FRAMES,
    ]

    if frame_matrix.shape[1] != (
        FRAME_FEATURES * N_FRAMES
    ):
        raise ValueError(
            "Invalid H1-H3 frame representation."
        )

    frame_matrix = np.asarray(
        frame_matrix,
        dtype=np.float32,
    ).reshape(
        n_samples,
        N_FRAMES,
        FRAME_FEATURES,
    )

    h3_sequence = (
        frame_matrix[:, :, -H3_PER_FRAME:]
    )

    expected_shape = (
        n_samples,
        N_FRAMES,
        H3_PER_FRAME,
    )

    if h3_sequence.shape != expected_shape:
        raise ValueError(
            f"Unexpected H3 shape: "
            f"{h3_sequence.shape}"
        )

    return h3_sequence


def build_h9_features(X):

    print(
        "\nConstructing H9 higher-order "
        "temporal features..."
    )

    h3_sequence = extract_h3_sequence(X)

    first_diff = np.diff(
        h3_sequence,
        axis=1,
    )

    mean_abs_first = np.mean(
        np.abs(first_diff),
        axis=1,
    )

    max_abs_first = np.max(
        np.abs(first_diff),
        axis=1,
    )

    second_diff = np.diff(
        first_diff,
        axis=1,
    )

    mean_abs_second = np.mean(
        np.abs(second_diff),
        axis=1,
    )

    max_abs_second = np.max(
        np.abs(second_diff),
        axis=1,
    )

    h9_features = np.concatenate(
        [
            mean_abs_first,
            max_abs_first,
            mean_abs_second,
            max_abs_second,
        ],
        axis=1,
    )

    if h9_features.shape[1] != H9_ADDITIONAL:
        raise ValueError(
            f"Expected {H9_ADDITIONAL} H9 features, "
            f"got {h9_features.shape[1]}"
        )

    return np.concatenate(
        [
            np.asarray(
                X,
                dtype=np.float32,
            ),
            h9_features.astype(np.float32),
        ],
        axis=1,
    )


# ============================================================
# H8 — FEATURE SELECTION
# ============================================================

def build_xgb_classifier():

    return XGBClassifier(
        **XGB_PARAMS
    )


def rank_features(
    X_train,
    y_train,
):

    print(
        "  H8: training feature-ranking model..."
    )

    model = build_xgb_classifier()

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    importance = model.feature_importances_

    ranking = np.argsort(
        importance
    )[::-1]

    return model, importance, ranking


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

    return np.asarray(
        X[
            :,
            AUX_FEATURE_START:AUX_FEATURE_END,
        ],
        dtype=np.float32,
    )


def crossfit_distance(
    X,
    distance,
    days,
    training_days,
):

    training_days = list(
        training_days
    )

    midpoint = len(training_days) // 2

    group_a_days = training_days[:midpoint]
    group_b_days = training_days[midpoint:]

    group_a = np.concatenate(
        [
            np.where(days == d)[0]
            for d in group_a_days
        ]
    )

    group_b = np.concatenate(
        [
            np.where(days == d)[0]
            for d in group_b_days
        ]
    )

    predictions = np.empty(
        len(X),
        dtype=np.float32,
    )

    X_aux = make_auxiliary_features(X)

    # A -> B

    model_a = build_ridge()

    model_a.fit(
        X_aux[group_a],
        distance[group_a],
    )

    predictions[group_b] = (
        model_a.predict(
            X_aux[group_b]
        ).astype(np.float32)
    )

    del model_a

    # B -> A

    model_b = build_ridge()

    model_b.fit(
        X_aux[group_b],
        distance[group_b],
    )

    predictions[group_a] = (
        model_b.predict(
            X_aux[group_a]
        ).astype(np.float32)
    )

    del model_b

    return predictions


def fit_distance_model_and_predict(
    X,
    distance,
    training_indices,
    prediction_indices,
):

    X_aux = make_auxiliary_features(X)

    model = build_ridge()

    model.fit(
        X_aux[training_indices],
        distance[training_indices],
    )

    predictions = model.predict(
        X_aux[prediction_indices]
    ).astype(np.float32)

    del model

    return predictions


# ============================================================
# H6 — CALIBRATION
# ============================================================

def class1_to_threat_probability(
    class1_probability
):

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


def select_threshold(
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

        if (
            best is None
            or candidate < best[0]
        ):

            best = (
                candidate,
                threshold,
                recall,
                false_alerts,
            )

    if best is None:

        threshold = float(
            THRESHOLD_GRID[0]
        )

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

        return (
            threshold,
            recall,
            false_alerts,
        )

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
):

    decisions = np.asarray(
        threat_decisions,
        dtype=bool,
    )

    days = extract_days(dt)

    output = np.zeros(
        len(decisions),
        dtype=bool,
    )

    for day in np.unique(days):

        indices = np.where(
            days == day
        )[0]

        day_decisions = (
            decisions[indices]
        )

        for local_i in range(
            len(indices)
        ):

            start = max(
                0,
                local_i
                - PERSISTENCE_WINDOW
                + 1,
            )

            recent = day_decisions[
                start:local_i + 1
            ]

            if (
                np.sum(recent)
                >= PERSISTENCE_REQUIRED
            ):

                output[
                    indices[local_i]
                ] = True

    return output


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    prediction,
    raw_probability,
    calibrated_probability,
    threshold,
):

    threat_true = (
        y_true == THREAT_CLASS
    ).astype(np.int8)

    prediction = np.asarray(
        prediction,
        dtype=np.int8,
    )

    cm = confusion_matrix(
        threat_true,
        prediction,
        labels=[1, 0],
    )

    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    precision = precision_score(
        threat_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        threat_true,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        threat_true,
        prediction,
        zero_division=0,
    )

    non_threat_true = 1 - threat_true
    non_threat_prediction = 1 - prediction

    non_threat_f1 = f1_score(
        non_threat_true,
        non_threat_prediction,
        zero_division=0,
    )

    macro_f1 = (
        f1 + non_threat_f1
    ) / 2.0

    weighted_f1 = f1_score(
        threat_true,
        prediction,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                threat_true,
                prediction,
            )
        ),
        "threat_precision": float(
            precision
        ),
        "threat_recall": float(
            recall
        ),
        "threat_f1": float(
            f1
        ),
        "non_threat_f1": float(
            non_threat_f1
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "roc_auc": float(
            roc_auc_score(
                threat_true,
                raw_probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                threat_true,
                raw_probability,
            )
        ),
        "brier_raw": float(
            brier_score_loss(
                threat_true,
                raw_probability,
            )
        ),
        "brier_calibrated": float(
            brier_score_loss(
                threat_true,
                calibrated_probability,
            )
        ),
        "threshold": float(
            threshold
        ),
        "true_threats": int(
            tp + fn
        ),
        "correct_threats": int(
            tp
        ),
        "missed_threats": int(
            fn
        ),
        "false_proximity_alerts": int(
            fp
        ),
        "correct_non_threats": int(
            tn
        ),
        "total": int(
            len(y_true)
        ),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    print("=" * 80)
    print("FINAL H1-H9 DAS PHYSICAL THREAT DETECTION EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X_base, y, dt = load_cached_data()

    raw_distance, raw_dt = (
        load_raw_distance()
    )

    distance = (
        align_raw_distance_to_cache(
            raw_distance,
            raw_dt,
            dt,
        )
    )

    days = extract_days(dt)

    unique_days = list(
        np.sort(
            np.unique(days)
        )
    )

    print(
        f"\nEvaluation days: "
        f"{len(unique_days)}"
    )

    if len(unique_days) != 10:
        raise ValueError(
            "Expected exactly 10 evaluation days."
        )

    # --------------------------------------------------------
    # BUILD H9
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("BUILDING FINAL H1-H4 + H9 REPRESENTATION")
    print("=" * 80)

    build_start = time.time()

    X = build_h9_features(
        X_base
    )

    print(
        f"Final pre-H8 representation: "
        f"{X.shape}"
    )

    if X.shape[1] != BASE_FEATURES:
        raise ValueError(
            f"Expected {BASE_FEATURES} features, "
            f"got {X.shape[1]}"
        )

    print(
        f"Representation build time: "
        f"{time.time() - build_start:.1f} s"
    )

    del X_base
    gc.collect()

    # --------------------------------------------------------
    # GLOBAL CONTAINERS
    # --------------------------------------------------------

    fold_results = []
    prediction_frames = []
    importance_records = []

    # --------------------------------------------------------
    # OUTER 10-FOLD LOOP
    # --------------------------------------------------------

    for fold_number, test_day in enumerate(
        unique_days,
        start=1,
    ):

        fold_start = time.time()

        print("\n" + "=" * 80)
        print(
            f"OUTER FOLD {fold_number}/10"
        )
        print(
            f"HELD-OUT TEST DAY: {test_day}"
        )
        print("=" * 80)

        training_days = [
            d
            for d in unique_days
            if d != test_day
        ]

        test_indices = np.where(
            days == test_day
        )[0]

        training_indices = np.concatenate(
            [
                np.where(days == d)[0]
                for d in training_days
            ]
        )

        # ====================================================
        # INNER VALIDATION DAY
        # ====================================================

        # Fixed deterministic choice:
        # final outer-training day becomes inner validation.
        validation_day = training_days[-1]

        inner_training_days = (
            training_days[:-1]
        )

        validation_indices = np.where(
            days == validation_day
        )[0]

        inner_training_indices = np.concatenate(
            [
                np.where(days == d)[0]
                for d in inner_training_days
            ]
        )

        print(
            f"Outer training samples: "
            f"{len(training_indices)}"
        )

        print(
            f"Inner training samples: "
            f"{len(inner_training_indices)}"
        )

        print(
            f"Inner validation day: "
            f"{validation_day}"
        )

        print(
            f"Inner validation samples: "
            f"{len(validation_indices)}"
        )

        print(
            f"Outer test samples: "
            f"{len(test_indices)}"
        )

        # ====================================================
        # H5 — INNER DISTANCE CROSS-FITTING
        # ====================================================

        print("\n--- H5 INNER DISTANCE CROSS-FITTING ---")

        inner_distance_predictions = (
            crossfit_distance(
                X,
                distance,
                days,
                inner_training_days,
            )
        )

        # Train distance model only on the eight
        # inner-training days and predict inner validation.

        validation_distance_prediction = (
            fit_distance_model_and_predict(
                X,
                distance,
                inner_training_indices,
                validation_indices,
            )
        )

        # ====================================================
        # H8 — INNER FEATURE SELECTION
        # ====================================================

        print("\n--- H8 INNER FEATURE SELECTION ---")

        X_inner_train = np.asarray(
            X[
                inner_training_indices
            ],
            dtype=np.float32,
        )

        y_inner_train = y[
            inner_training_indices
        ]

        X_validation_full = np.asarray(
            X[
                validation_indices
            ],
            dtype=np.float32,
        )

        y_validation = y[
            validation_indices
        ]

        importance_model, importance, ranking = (
            rank_features(
                X_inner_train,
                y_inner_train,
            )
        )

        ranking_path = os.path.join(
            RESULTS_DIR,
            "feature_importance",
        )

        os.makedirs(
            ranking_path,
            exist_ok=True,
        )

        ranking_df = pd.DataFrame(
            {
                "feature_index": ranking,
                "importance": importance[
                    ranking
                ],
            }
        )

        ranking_df.to_csv(
            os.path.join(
                ranking_path,
                f"fold_{fold_number:02d}_{test_day}.csv",
            ),
            index=False,
        )

        importance_records.append(
            ranking_df.head(TOP_K)
        )

        selected_inner = ranking[
            :TOP_K
        ]

        # ====================================================
        # INNER H5 FEATURES
        # ====================================================

        X_inner_train_h5 = np.column_stack(
            [
                X_inner_train[
                    :,
                    selected_inner,
                ],
                inner_distance_predictions[
                    inner_training_indices
                ],
            ]
        ).astype(np.float32)

        X_validation_h5 = np.column_stack(
            [
                X_validation_full[
                    :,
                    selected_inner,
                ],
                validation_distance_prediction,
            ]
        ).astype(np.float32)

        if (
            X_inner_train_h5.shape[1]
            != TOP_K + 1
        ):
            raise ValueError(
                "Invalid inner H5 feature dimension."
            )

        # ====================================================
        # H6 — INNER CALIBRATION
        # ====================================================

        print("\n--- H6 INNER CALIBRATION ---")

        validation_classifier = (
            build_xgb_classifier()
        )

        validation_classifier.fit(
            X_inner_train_h5,
            y_inner_train,
            verbose=False,
        )

        validation_class1_probability = (
            validation_classifier
            .predict_proba(
                X_validation_h5
            )[:, 1]
        )

        validation_raw_threat_probability = (
            class1_to_threat_probability(
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
        ) = select_threshold(
            validation_calibrated_probability,
            y_validation,
        )

        print(
            f"Selected threshold: "
            f"{threshold:.3f}"
        )

        print(
            f"Validation threat recall: "
            f"{validation_threat_recall:.4f}"
        )

        print(
            f"Validation false alerts: "
            f"{validation_false_alerts}"
        )

        del validation_classifier

        # ====================================================
        # FINAL OUTER H5 CROSS-FITTING
        # ====================================================

        print(
            "\n--- H5 FINAL OUTER CROSS-FITTING ---"
        )

        outer_distance_predictions = (
            crossfit_distance(
                X,
                distance,
                days,
                training_days,
            )
        )

        outer_test_distance_prediction = (
            fit_distance_model_and_predict(
                X,
                distance,
                training_indices,
                test_indices,
            )
        )

        # ====================================================
        # FINAL OUTER H8
        # ====================================================

        print(
            "\n--- H8 FINAL OUTER FEATURE SELECTION ---"
        )

        X_outer_train = np.asarray(
            X[
                training_indices
            ],
            dtype=np.float32,
        )

        y_outer_train = y[
            training_indices
        ]

        X_outer_test = np.asarray(
            X[
                test_indices
            ],
            dtype=np.float32,
        )

        y_outer_test = y[
            test_indices
        ]

        outer_importance_model, outer_importance, outer_ranking = (
            rank_features(
                X_outer_train,
                y_outer_train,
            )
        )

        outer_ranking_df = pd.DataFrame(
            {
                "feature_index": outer_ranking,
                "importance": outer_importance[
                    outer_ranking
                ],
            }
        )

        outer_ranking_df.to_csv(
            os.path.join(
                ranking_path,
                f"fold_{fold_number:02d}_{test_day}_outer.csv",
            ),
            index=False,
        )

        selected_outer = outer_ranking[
            :TOP_K
        ]

        # ====================================================
        # FINAL OUTER CLASSIFIER
        # ====================================================

        X_outer_train_h5 = np.column_stack(
            [
                X_outer_train[
                    :,
                    selected_outer,
                ],
                outer_distance_predictions[
                    training_indices
                ],
            ]
        ).astype(np.float32)

        X_outer_test_h5 = np.column_stack(
            [
                X_outer_test[
                    :,
                    selected_outer,
                ],
                outer_test_distance_prediction,
            ]
        ).astype(np.float32)

        print(
            f"Final classifier train shape: "
            f"{X_outer_train_h5.shape}"
        )

        print(
            f"Final classifier test shape: "
            f"{X_outer_test_h5.shape}"
        )

        if (
            X_outer_train_h5.shape[1]
            != TOP_K + 1
        ):
            raise ValueError(
                "Invalid final feature dimension."
            )

        final_classifier = (
            build_xgb_classifier()
        )

        final_classifier.fit(
            X_outer_train_h5,
            y_outer_train,
            verbose=False,
        )

        # ====================================================
        # RAW PROBABILITIES
        # ====================================================

        class1_probability = (
            final_classifier
            .predict_proba(
                X_outer_test_h5
            )[:, 1]
        )

        raw_threat_probability = (
            class1_to_threat_probability(
                class1_probability
            )
        )

        # ====================================================
        # H6 CALIBRATION
        # ====================================================

        calibrated_threat_probability = (
            calibrator.predict(
                raw_threat_probability
            )
        )

        raw_threat_prediction = (
            raw_threat_probability
            >= threshold
        )

        # ====================================================
        # H7 PERSISTENCE
        # ====================================================

        persistent_threat_prediction = (
            apply_temporal_persistence(
                raw_threat_prediction,
                dt[
                    test_indices
                ],
            )
        )

        # ====================================================
        # METRICS — RAW DECISION
        # ====================================================

        raw_metrics = calculate_metrics(
            y_outer_test,
            raw_threat_prediction,
            raw_threat_probability,
            calibrated_threat_probability,
            threshold,
        )

        # ====================================================
        # METRICS — FINAL H7 DECISION
        # ====================================================

        final_metrics = calculate_metrics(
            y_outer_test,
            persistent_threat_prediction,
            raw_threat_probability,
            calibrated_threat_probability,
            threshold,
        )

        # ====================================================
        # STORE FOLD METRICS
        # ====================================================

        result = {
            "fold": fold_number,
            "test_day": test_day,
            "validation_day": validation_day,

            "n_features_before_h8": BASE_FEATURES,
            "n_features_after_h8": TOP_K,
            "n_final_classifier_features": TOP_K + 1,

            "threshold": threshold,

            "validation_threat_recall":
                validation_threat_recall,

            "validation_false_alerts":
                validation_false_alerts,

            "raw_accuracy":
                raw_metrics["accuracy"],

            "raw_threat_precision":
                raw_metrics["threat_precision"],

            "raw_threat_recall":
                raw_metrics["threat_recall"],

            "raw_threat_f1":
                raw_metrics["threat_f1"],

            "raw_roc_auc":
                raw_metrics["roc_auc"],

            "raw_pr_auc":
                raw_metrics["pr_auc"],

            "raw_brier":
                raw_metrics["brier_raw"],

            "calibrated_brier":
                raw_metrics["brier_calibrated"],

            "final_accuracy":
                final_metrics["accuracy"],

            "final_threat_precision":
                final_metrics["threat_precision"],

            "final_threat_recall":
                final_metrics["threat_recall"],

            "final_threat_f1":
                final_metrics["threat_f1"],

            "final_non_threat_f1":
                final_metrics["non_threat_f1"],

            "final_macro_f1":
                final_metrics["macro_f1"],

            "final_weighted_f1":
                final_metrics["weighted_f1"],

            "final_roc_auc":
                final_metrics["roc_auc"],

            "final_pr_auc":
                final_metrics["pr_auc"],

            "final_brier_raw":
                final_metrics["brier_raw"],

            "final_brier_calibrated":
                final_metrics["brier_calibrated"],

            "false_proximity_alerts":
                final_metrics[
                    "false_proximity_alerts"
                ],

            "missed_threats":
                final_metrics[
                    "missed_threats"
                ],

            "correct_threats":
                final_metrics[
                    "correct_threats"
                ],

            "correct_non_threats":
                final_metrics[
                    "correct_non_threats"
                ],

            "total":
                final_metrics["total"],

            "tp":
                final_metrics["tp"],

            "fn":
                final_metrics["fn"],

            "fp":
                final_metrics["fp"],

            "tn":
                final_metrics["tn"],

            "fold_runtime_seconds":
                time.time() - fold_start,
        }

        fold_results.append(
            result
        )

        # ====================================================
        # STORE SAMPLE-LEVEL PREDICTIONS
        # ====================================================

        fold_prediction_df = pd.DataFrame(
            {
                "datetime": [
                    str(x)
                    for x in dt[
                        test_indices
                    ]
                ],
                "test_day": test_day,
                "fold": fold_number,

                "true_label": y_outer_test,

                # Evaluation metadata only.
                "distance_m":
                    distance[
                        test_indices
                    ],

                "raw_threat_probability":
                    raw_threat_probability,

                "calibrated_threat_probability":
                    calibrated_threat_probability,

                "threshold":
                    threshold,

                "raw_threat_prediction":
                    raw_threat_prediction.astype(
                        np.int8
                    ),

                "final_persistent_prediction":
                    persistent_threat_prediction.astype(
                        np.int8
                    ),
            }
        )

        prediction_frames.append(
            fold_prediction_df
        )

        # ====================================================
        # PRINT FOLD RESULT
        # ====================================================

        print("\nFINAL FOLD RESULT")
        print(
            f"Accuracy:              "
            f"{final_metrics['accuracy']:.4f}"
        )
        print(
            f"Threat precision:      "
            f"{final_metrics['threat_precision']:.4f}"
        )
        print(
            f"Threat recall:         "
            f"{final_metrics['threat_recall']:.4f}"
        )
        print(
            f"Threat F1:             "
            f"{final_metrics['threat_f1']:.4f}"
        )
        print(
            f"Non-threat F1:         "
            f"{final_metrics['non_threat_f1']:.4f}"
        )
        print(
            f"Macro F1:              "
            f"{final_metrics['macro_f1']:.4f}"
        )
        print(
            f"ROC-AUC:               "
            f"{final_metrics['roc_auc']:.4f}"
        )
        print(
            f"PR-AUC:                "
            f"{final_metrics['pr_auc']:.4f}"
        )
        print(
            f"Brier raw:             "
            f"{final_metrics['brier_raw']:.6f}"
        )
        print(
            f"Brier calibrated:      "
            f"{final_metrics['brier_calibrated']:.6f}"
        )
        print(
            f"False proximity alerts:"
            f" {final_metrics['false_proximity_alerts']}"
        )
        print(
            f"Missed threats:        "
            f"{final_metrics['missed_threats']}"
        )
        print(
            f"Fold runtime:          "
            f"{time.time() - fold_start:.1f} s"
        )

        # ====================================================
        # MEMORY CLEANUP
        # ====================================================

        del (
            X_inner_train,
            X_validation_full,
            X_outer_train,
            X_outer_test,
            X_inner_train_h5,
            X_validation_h5,
            X_outer_train_h5,
            X_outer_test_h5,
            importance_model,
            importance,
            ranking,
            outer_importance_model,
            outer_importance,
            outer_ranking,
            final_classifier,
            calibrator,
            inner_distance_predictions,
            validation_distance_prediction,
            outer_distance_predictions,
            outer_test_distance_prediction,
        )

        gc.collect()

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    fold_metrics_path = os.path.join(
        RESULTS_DIR,
        "final_fold_metrics.csv",
    )

    fold_df.to_csv(
        fold_metrics_path,
        index=False,
    )

    # ========================================================
    # SAVE SAMPLE PREDICTIONS
    # ========================================================

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        "final_predictions.csv",
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # ========================================================
    # POOLED METRICS
    # ========================================================

    y_true = (
        predictions_df[
            "true_label"
        ].to_numpy()
    )

    raw_probability = (
        predictions_df[
            "raw_threat_probability"
        ].to_numpy()
    )

    calibrated_probability = (
        predictions_df[
            "calibrated_threat_probability"
        ].to_numpy()
    )

    final_prediction = (
        predictions_df[
            "final_persistent_prediction"
        ].to_numpy()
    )

    pooled_metrics = calculate_metrics(
        y_true,
        final_prediction,
        raw_probability,
        calibrated_probability,
        float(
            predictions_df[
                "threshold"
            ].mean()
        ),
    )

    # ========================================================
    # PER-DAY METRICS
    # ========================================================

    per_day_results = []

    for day in unique_days:

        mask = (
            predictions_df[
                "test_day"
            ].to_numpy()
            == day
        )

        day_metrics = calculate_metrics(
            y_true[mask],
            final_prediction[mask],
            raw_probability[mask],
            calibrated_probability[mask],
            float(
                predictions_df.loc[
                    mask,
                    "threshold",
                ].iloc[0]
            ),
        )

        day_metrics["test_day"] = day
        day_metrics["fold"] = int(
            predictions_df.loc[
                mask,
                "fold",
            ].iloc[0]
        )

        per_day_results.append(
            day_metrics
        )

    per_day_df = pd.DataFrame(
        per_day_results
    )

    per_day_path = os.path.join(
        RESULTS_DIR,
        "final_per_day_metrics.csv",
    )

    per_day_df.to_csv(
        per_day_path,
        index=False,
    )

    # ========================================================
    # POOLED CONFUSION MATRIX
    # ========================================================

    threat_true = (
        y_true == THREAT_CLASS
    ).astype(np.int8)

    cm = confusion_matrix(
        threat_true,
        final_prediction,
        labels=[1, 0],
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "experiment":
            "FINAL_H1_H9",

        "evaluation":
            "10-fold leave-one-day-out",

        "n_samples":
            int(len(predictions_df)),

        "n_days":
            int(len(unique_days)),

        "pre_h8_features":
            int(BASE_FEATURES),

        "h8_top_k":
            int(TOP_K),

        "final_classifier_features":
            int(TOP_K + 1),

        "h5_auxiliary":
            "fold-contained Ridge predicted distance",

        "h6_calibration":
            "inner-validation isotonic regression",

        "h6_min_threat_recall":
            MIN_THREAT_RECALL,

        "h7_persistence_window":
            PERSISTENCE_WINDOW,

        "h7_persistence_required":
            PERSISTENCE_REQUIRED,

        "accuracy":
            pooled_metrics["accuracy"],

        "threat_precision":
            pooled_metrics["threat_precision"],

        "threat_recall":
            pooled_metrics["threat_recall"],

        "threat_f1":
            pooled_metrics["threat_f1"],

        "non_threat_f1":
            pooled_metrics["non_threat_f1"],

        "macro_f1":
            pooled_metrics["macro_f1"],

        "weighted_f1":
            pooled_metrics["weighted_f1"],

        "roc_auc":
            pooled_metrics["roc_auc"],

        "pr_auc":
            pooled_metrics["pr_auc"],

        "brier_raw":
            pooled_metrics["brier_raw"],

        "brier_calibrated":
            pooled_metrics["brier_calibrated"],

        "false_proximity_alerts":
            pooled_metrics[
                "false_proximity_alerts"
            ],

        "missed_threats":
            pooled_metrics[
                "missed_threats"
            ],

        "correct_threats":
            pooled_metrics[
                "correct_threats"
            ],

        "correct_non_threats":
            pooled_metrics[
                "correct_non_threats"
            ],

        "confusion_matrix_labels":
            [
                "threat",
                "non_threat",
            ],

        "confusion_matrix":
            cm.tolist(),

        "mean_fold_accuracy":
            float(
                fold_df[
                    "final_accuracy"
                ].mean()
            ),

        "std_fold_accuracy":
            float(
                fold_df[
                    "final_accuracy"
                ].std()
            ),

        "mean_fold_threat_precision":
            float(
                fold_df[
                    "final_threat_precision"
                ].mean()
            ),

        "std_fold_threat_precision":
            float(
                fold_df[
                    "final_threat_precision"
                ].std()
            ),

        "mean_fold_threat_recall":
            float(
                fold_df[
                    "final_threat_recall"
                ].mean()
            ),

        "std_fold_threat_recall":
            float(
                fold_df[
                    "final_threat_recall"
                ].std()
            ),

        "mean_fold_threat_f1":
            float(
                fold_df[
                    "final_threat_f1"
                ].mean()
            ),

        "std_fold_threat_f1":
            float(
                fold_df[
                    "final_threat_f1"
                ].std()
            ),

        "mean_fold_auc":
            float(
                fold_df[
                    "final_roc_auc"
                ].mean()
            ),

        "std_fold_auc":
            float(
                fold_df[
                    "final_roc_auc"
                ].std()
            ),

        "mean_fold_pr_auc":
            float(
                fold_df[
                    "final_pr_auc"
                ].mean()
            ),

        "std_fold_pr_auc":
            float(
                fold_df[
                    "final_pr_auc"
                ].std()
            ),

        "mean_threshold":
            float(
                fold_df[
                    "threshold"
                ].mean()
            ),

        "std_threshold":
            float(
                fold_df[
                    "threshold"
                ].std()
            ),
    }

    summary_path = os.path.join(
        RESULTS_DIR,
        "final_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL PRINT
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL H1-H9 EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"Samples:                 "
        f"{summary['n_samples']}"
    )

    print(
        f"Accuracy:                "
        f"{summary['accuracy']:.4f}"
    )

    print(
        f"Threat precision:        "
        f"{summary['threat_precision']:.4f}"
    )

    print(
        f"Threat recall:           "
        f"{summary['threat_recall']:.4f}"
    )

    print(
        f"Threat F1:               "
        f"{summary['threat_f1']:.4f}"
    )

    print(
        f"Non-threat F1:           "
        f"{summary['non_threat_f1']:.4f}"
    )

    print(
        f"Macro F1:                "
        f"{summary['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:             "
        f"{summary['weighted_f1']:.4f}"
    )

    print(
        f"ROC-AUC:                 "
        f"{summary['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:                  "
        f"{summary['pr_auc']:.4f}"
    )

    print(
        f"Brier raw:               "
        f"{summary['brier_raw']:.6f}"
    )

    print(
        f"Brier calibrated:        "
        f"{summary['brier_calibrated']:.6f}"
    )

    print(
        f"False proximity alerts:  "
        f"{summary['false_proximity_alerts']}"
    )

    print(
        f"Missed threats:          "
        f"{summary['missed_threats']}"
    )

    print("\nPooled confusion matrix:")
    print(cm)

    print("\nSaved:")
    print(
        f"  {fold_metrics_path}"
    )
    print(
        f"  {per_day_path}"
    )
    print(
        f"  {predictions_path}"
    )
    print(
        f"  {summary_path}"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_experiment()
