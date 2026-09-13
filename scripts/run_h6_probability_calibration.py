#!/usr/bin/env python3

"""
H6 — Probability Calibration and Threat-Aware Decision

Hypothesis:
    Calibrating classifier probabilities and selecting a threat-aware
    decision threshold can improve the reliability of physical-threat
    alerts, particularly by reducing false alarms while maintaining
    high proximity-threat recall.

Important:
    H6 is a decision/calibration layer.
    It does NOT modify the frozen H1-H4 representation.

Frozen representation:
    H1 + H2 + H3 + H4 = 2800 features

Threat class:
    class 0 = vessel distance <= 1000 m
    class 1 = vessel distance > 1000 m

Therefore:
    threat_probability = 1 - P(class 1)

Leakage control:
    Calibration and threshold selection must use validation data only.
    The outer test day must never be used for calibration or threshold
    selection.

Current stage:
    --sanity validates the implementation.
    Full H6 evaluation is intentionally deferred until the cumulative
    H5 + H6 + H7 experiment.
"""

import os
import sys

import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================================
# PATHS
# ============================================================================

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPECTED_FEATURES = 2800

THREAT_CLASS = 0

THREAT_THRESHOLD = 1000.0

# Operational requirement used for H6 threshold selection.
#
# The threshold-selection procedure seeks a threshold that maintains
# at least 90% recall for the physical-threat class while minimizing
# false proximity alerts.
MIN_THREAT_RECALL = 0.90

# Candidate threat-probability thresholds.
THRESHOLD_GRID = np.linspace(
    0.01,
    0.99,
    99,
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cached_data():
    """
    Load the frozen H1-H4 representation.

    The feature matrix is memory-mapped because the 2800-feature
    representation is large.

    Returns
    -------
    X : np.ndarray / np.memmap
        Frozen H1-H4 features.

    y : np.ndarray
        Binary threat targets.

    dt : np.ndarray
        Datetimes corresponding to each reduced sample.
    """

    print("Loading cached H1-H4 representation...")

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(
            f"Feature cache not found: {FEATURE_PATH}"
        )

    if not os.path.exists(TARGET_PATH):
        raise FileNotFoundError(
            f"Target cache not found: {TARGET_PATH}"
        )

    if not os.path.exists(DATETIME_PATH):
        raise FileNotFoundError(
            f"Datetime cache not found: {DATETIME_PATH}"
        )

    # Memory-map the large feature matrix.
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

    print(f"Cached features: {X.shape}")
    print(f"Binary targets:  {y.shape}")
    print(f"Datetimes:       {dt.shape}")

    # ------------------------------------------------------------------------
    # Dimensionality checks
    # ------------------------------------------------------------------------

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, got shape {X.shape}"
        )

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {X.shape[1]}"
        )

    # ------------------------------------------------------------------------
    # Alignment checks
    # ------------------------------------------------------------------------

    if len(X) != len(y):
        raise ValueError(
            f"X/y length mismatch: {len(X)} vs {len(y)}"
        )

    if len(X) != len(dt):
        raise ValueError(
            f"X/datetime length mismatch: "
            f"{len(X)} vs {len(dt)}"
        )

    # ------------------------------------------------------------------------
    # Target checks
    # ------------------------------------------------------------------------

    y = np.asarray(y)

    unique_targets = np.unique(y)

    if not np.all(
        np.isin(unique_targets, [0, 1])
    ):
        raise ValueError(
            f"Expected binary targets {{0,1}}, "
            f"got {unique_targets}"
        )

    # ------------------------------------------------------------------------
    # Feature finite-value check
    # ------------------------------------------------------------------------

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains non-finite values."
        )

    print("H6 cache dimensionality/alignment: PASS")

    return (
        X,
        y.astype(np.int8),
        np.asarray(dt),
    )


# ============================================================================
# PROBABILITY CONVERSION
# ============================================================================

def class1_to_threat_probability(class1_probability):
    """
    Convert XGBoost class-1 probability into physical-threat
    probability.

    Dataset convention:
        class 0 = distance <= 1000 m = physical threat
        class 1 = distance > 1000 m = non-threat

    Therefore:

        P(threat) = 1 - P(class 1)
    """

    class1_probability = np.asarray(
        class1_probability,
        dtype=np.float64,
    )

    threat_probability = 1.0 - class1_probability

    return np.clip(
        threat_probability,
        0.0,
        1.0,
    )


# ============================================================================
# CALIBRATION
# ============================================================================

def fit_isotonic_calibrator(
    validation_threat_probability,
    validation_target,
):
    """
    Fit an isotonic probability calibrator using validation data only.

    Parameters
    ----------
    validation_threat_probability : array-like
        Raw threat probabilities.

    validation_target : array-like
        Binary target where 0 is threat and 1 is non-threat.

    Returns
    -------
    calibrator : IsotonicRegression
        Fitted isotonic regression model.
    """

    validation_threat_probability = np.asarray(
        validation_threat_probability,
        dtype=np.float64,
    )

    validation_target = np.asarray(
        validation_target,
        dtype=np.int8,
    )

    # IsotonicRegression expects the positive event to correspond
    # to increasing target values. Since our threat class is 0,
    # convert the target to:
    #
    #   1 = physical threat
    #   0 = non-threat
    #
    threat_target = (
        validation_target == THREAT_CLASS
    ).astype(np.int8)

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        validation_threat_probability,
        threat_target,
    )

    return calibrator


# ============================================================================
# THRESHOLD SELECTION
# ============================================================================

def select_threat_threshold(
    threat_probability,
    target,
    min_recall=MIN_THREAT_RECALL,
):
    """
    Select a threat-probability threshold using validation data only.

    Objective:
        1. Require threat recall >= min_recall.
        2. Among feasible thresholds, minimize false proximity alerts.
        3. Break ties using higher threat recall.

    Parameters
    ----------
    threat_probability : array-like
        Calibrated probability of physical threat.

    target : array-like
        Binary target:
            0 = threat
            1 = non-threat

    min_recall : float
        Minimum required threat recall.

    Returns
    -------
    best_threshold : float
        Selected probability threshold.

    best_stats : dict
        Validation statistics for the selected threshold.
    """

    threat_probability = np.asarray(
        threat_probability,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.int8,
    )

    actual_threat = (
        target == THREAT_CLASS
    )

    best_threshold = None
    best_stats = None

    for threshold in THRESHOLD_GRID:

        predicted_threat = (
            threat_probability >= threshold
        )

        # Confusion matrix for:
        #
        # actual threat / non-threat
        # predicted threat / non-threat
        #
        # Here we explicitly construct the quantities because
        # threat is class 0 in the original dataset.
        true_threat = np.sum(
            predicted_threat & actual_threat
        )

        missed_threat = np.sum(
            (~predicted_threat) & actual_threat
        )

        false_threat = np.sum(
            predicted_threat & (~actual_threat)
        )

        correct_non_threat = np.sum(
            (~predicted_threat) & (~actual_threat)
        )

        denominator = (
            true_threat + missed_threat
        )

        if denominator == 0:
            continue

        threat_recall = (
            true_threat / denominator
        )

        if threat_recall < min_recall:
            continue

        stats = {
            "threshold": float(threshold),
            "threat_recall": float(threat_recall),
            "false_proximity_alerts": int(false_threat),
            "missed_threats": int(missed_threat),
            "correct_threats": int(true_threat),
            "correct_non_threats": int(correct_non_threat),
        }

        if best_stats is None:
            best_threshold = float(threshold)
            best_stats = stats
            continue

        # Primary objective:
        # minimize false proximity alerts.
        #
        # Secondary objective:
        # maximize threat recall.
        if (
            stats["false_proximity_alerts"]
            < best_stats["false_proximity_alerts"]
        ):
            best_threshold = float(threshold)
            best_stats = stats

        elif (
            stats["false_proximity_alerts"]
            == best_stats["false_proximity_alerts"]
            and stats["threat_recall"]
            > best_stats["threat_recall"]
        ):
            best_threshold = float(threshold)
            best_stats = stats

    if best_threshold is None:
        raise RuntimeError(
            "No threshold satisfies the required threat recall "
            f"of {min_recall:.2f}."
        )

    return (
        best_threshold,
        best_stats,
    )


# ============================================================================
# OPERATIONAL METRICS
# ============================================================================

def calculate_threat_metrics(
    threat_probability,
    target,
    threshold,
):
    """
    Calculate operational physical-threat metrics.

    Threat definition:
        class 0 = distance <= 1000 m

    Prediction:
        threat_probability >= threshold
        => predicted physical threat
    """

    threat_probability = np.asarray(
        threat_probability,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.int8,
    )

    actual_threat = (
        target == THREAT_CLASS
    )

    predicted_threat = (
        threat_probability >= threshold
    )

    # Convert to conventional binary labels where:
    #
    #   1 = physical threat
    #   0 = non-threat
    #
    actual_binary = actual_threat.astype(np.int8)
    predicted_binary = predicted_threat.astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        actual_binary,
        predicted_binary,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(
                actual_binary,
                predicted_binary,
            )
        ),
        "threat_precision": float(
            precision_score(
                actual_binary,
                predicted_binary,
                zero_division=0,
            )
        ),
        "threat_recall": float(
            recall_score(
                actual_binary,
                predicted_binary,
                zero_division=0,
            )
        ),
        "threat_f1": float(
            f1_score(
                actual_binary,
                predicted_binary,
                zero_division=0,
            )
        ),
        "false_proximity_alerts": int(fp),
        "missed_threats": int(fn),
        "correct_threats": int(tp),
        "correct_non_threats": int(tn),
        "threshold": float(threshold),
    }

    return metrics


# ============================================================================
# SANITY CHECK
# ============================================================================

def sanity_check():
    """
    Validate the H6 implementation without training a full model.

    The sanity test:
        1. Loads the real frozen H1-H4 cache.
        2. Verifies 2800-dimensional representation.
        3. Creates synthetic classifier probabilities.
        4. Converts class-1 probability to threat probability.
        5. Fits isotonic calibration on synthetic validation data.
        6. Selects an operational threat threshold.
        7. Computes threat-aware metrics.
    """

    print()
    print("=" * 70)
    print("H6 PROBABILITY CALIBRATION SANITY CHECK")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load frozen representation
    # ------------------------------------------------------------------------

    X, y, dt = load_cached_data()

    print()
    print("Frozen H1-H4 representation:")
    print(f"  Features: {X.shape}")
    print(f"  Targets:  {y.shape}")
    print(f"  Datetimes: {dt.shape}")

    # ------------------------------------------------------------------------
    # Basic data checks
    # ------------------------------------------------------------------------

    assert X.shape[1] == EXPECTED_FEATURES
    assert len(X) == len(y)
    assert len(X) == len(dt)

    print()
    print("Frozen representation checks: PASS")

    # ------------------------------------------------------------------------
    # Synthetic probability data
    # ------------------------------------------------------------------------
    #
    # We deliberately use a small deterministic synthetic sample.
    # The real classifier is NOT trained here.
    # This is only a logic/integration test.
    #

    rng = np.random.default_rng(42)

    n_synthetic = 1000

    synthetic_target = np.tile(
        np.array([0, 1], dtype=np.int8),
        n_synthetic // 2,
    )

    # Physical threat:
    # higher threat probability.
    #
    # Non-threat:
    # lower threat probability.
    #

    synthetic_threat_probability = np.empty(
        n_synthetic,
        dtype=np.float64,
    )

    threat_mask = (
        synthetic_target == THREAT_CLASS
    )

    synthetic_threat_probability[
        threat_mask
    ] = rng.uniform(
        0.55,
        0.95,
        size=np.sum(threat_mask),
    )

    synthetic_threat_probability[
        ~threat_mask
    ] = rng.uniform(
        0.05,
        0.45,
        size=np.sum(~threat_mask),
    )

    # Convert threat probability into class-1 probability.
    synthetic_class1_probability = (
        1.0 - synthetic_threat_probability
    )

    # ------------------------------------------------------------------------
    # Verify probability conversion
    # ------------------------------------------------------------------------

    recovered_threat_probability = (
        class1_to_threat_probability(
            synthetic_class1_probability
        )
    )

    if not np.allclose(
        recovered_threat_probability,
        synthetic_threat_probability,
    ):
        raise AssertionError(
            "Class-1 to threat probability conversion failed."
        )

    print("Probability conversion: PASS")

    # ------------------------------------------------------------------------
    # Split synthetic validation data
    # ------------------------------------------------------------------------

    split = n_synthetic // 2

    validation_probability = (
        synthetic_threat_probability[:split]
    )

    validation_target = (
        synthetic_target[:split]
    )

    evaluation_probability = (
        synthetic_threat_probability[split:]
    )

    evaluation_target = (
        synthetic_target[split:]
    )

    # ------------------------------------------------------------------------
    # Fit calibration model
    # ------------------------------------------------------------------------

    calibrator = fit_isotonic_calibrator(
        validation_probability,
        validation_target,
    )

    calibrated_validation_probability = (
        calibrator.predict(
            validation_probability
        )
    )

    calibrated_evaluation_probability = (
        calibrator.predict(
            evaluation_probability
        )
    )

    if not np.all(
        np.isfinite(
            calibrated_validation_probability
        )
    ):
        raise AssertionError(
            "Calibrated validation probabilities contain "
            "non-finite values."
        )

    if not np.all(
        np.isfinite(
            calibrated_evaluation_probability
        )
    ):
        raise AssertionError(
            "Calibrated evaluation probabilities contain "
            "non-finite values."
        )

    print("Isotonic calibration: PASS")

    # ------------------------------------------------------------------------
    # Threshold selection
    # ------------------------------------------------------------------------

    selected_threshold, threshold_stats = (
        select_threat_threshold(
            calibrated_validation_probability,
            validation_target,
            min_recall=MIN_THREAT_RECALL,
        )
    )

    print()
    print("Threshold selection: PASS")
    print(
        f"  Selected threshold: "
        f"{selected_threshold:.3f}"
    )
    print(
        f"  Validation threat recall: "
        f"{threshold_stats['threat_recall']:.4f}"
    )
    print(
        f"  Validation false proximity alerts: "
        f"{threshold_stats['false_proximity_alerts']}"
    )

    # ------------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------------

    evaluation_metrics = calculate_threat_metrics(
        calibrated_evaluation_probability,
        evaluation_target,
        selected_threshold,
    )

    print()
    print("Evaluation metric calculation: PASS")
    print(
        f"  Accuracy: "
        f"{evaluation_metrics['accuracy']:.4f}"
    )
    print(
        f"  Threat precision: "
        f"{evaluation_metrics['threat_precision']:.4f}"
    )
    print(
        f"  Threat recall: "
        f"{evaluation_metrics['threat_recall']:.4f}"
    )
    print(
        f"  Threat F1: "
        f"{evaluation_metrics['threat_f1']:.4f}"
    )

    # ------------------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------------------

    if not (
        0.0
        <= evaluation_metrics["threat_recall"]
        <= 1.0
    ):
        raise AssertionError(
            "Threat recall outside [0,1]."
        )

    if not (
        0.0
        <= evaluation_metrics["threat_precision"]
        <= 1.0
    ):
        raise AssertionError(
            "Threat precision outside [0,1]."
        )

    if not (
        0.0
        <= evaluation_metrics["threat_f1"]
        <= 1.0
    ):
        raise AssertionError(
            "Threat F1 outside [0,1]."
        )

    print()
    print("=" * 70)
    print("H6 SANITY CHECK: PASS")
    print("=" * 70)
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point.
    """

    if len(sys.argv) != 2 or sys.argv[1] != "--sanity":
        print(
            "Usage:\n"
            "  python scripts/run_h6_probability_calibration.py "
            "--sanity"
        )
        print()
        print(
            "Full H6 evaluation is intentionally deferred until "
            "the cumulative H5 + H6 + H7 experiment."
        )
        sys.exit(1)

    sanity_check()


if __name__ == "__main__":
    main()
