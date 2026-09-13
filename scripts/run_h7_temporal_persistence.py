#!/usr/bin/env python3

"""
H7 — Temporal Persistence / Event Decision Layer

Hypothesis:
    Requiring physical-threat predictions to persist across consecutive
    observations can suppress isolated false alarms while preserving
    sustained vessel-threat events.

H7 is a temporal decision layer.

It does NOT modify the frozen H1-H4 representation.

Pipeline:

    H1-H4 features
          |
          v
    classifier probabilities
          |
          v
    H6 probability calibration
          |
          v
    threat probability
          |
          v
    H6 threat-aware threshold
          |
          v
    H7 temporal persistence
          |
          v
    final physical-threat decision

Dataset convention:
    class 0 = vessel distance <= 1000 m = physical proximity/threat
    class 1 = vessel distance > 1000 m = non-proximity

Important:
    H7 is evaluated only on chronological sequences within each day.
    Persistence is never allowed to cross a day boundary.

Current stage:
    --sanity validates the implementation.
    Full H7 evaluation is intentionally deferred until the cumulative
    H5 + H6 + H7 experiment.
"""

import os
import sys

import numpy as np


# ============================================================================
# REPOSITORY PATHS
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

THREAT_DISTANCE_METERS = 1000.0

# H7 persistence configuration.
#
# A prediction is considered persistent if at least
# PERSISTENCE_REQUIRED observations in the most recent
# PERSISTENCE_WINDOW observations indicate threat.
#
# Default:
#
#     2 of the last 3 observations
#
# This is deliberately different from simply changing the
# original model's 1/3/5 majority-voting parameter.

PERSISTENCE_WINDOW = 3

PERSISTENCE_REQUIRED = 2


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cached_data():
    """
    Load the frozen H1-H4 representation.

    Returns
    -------
    X : np.ndarray
        Frozen H1-H4 feature matrix.

    y : np.ndarray
        Binary physical-threat target.

    dt : np.ndarray
        Datetimes corresponding to the reduced observations.
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

    # The feature matrix is large, so use memory mapping.
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
    # Feature checks
    # ------------------------------------------------------------------------

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix, got {X.shape}"
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
    # Datetime checks
    # ------------------------------------------------------------------------

    if len(dt) == 0:
        raise ValueError(
            "Datetime array is empty."
        )

    # ------------------------------------------------------------------------
    # Feature finite-value check
    # ------------------------------------------------------------------------

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains non-finite values."
        )

    print("H7 cache dimensionality/alignment: PASS")

    return (
        X,
        y.astype(np.int8),
        np.asarray(dt),
    )


# ============================================================================
# DATETIME NORMALIZATION
# ============================================================================

def normalize_datetime(value):
    """
    Convert a datetime-like value into a string representation.

    This function deliberately keeps the original datetime information
    intact while providing a stable representation for day grouping.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def get_day_string(value):
    """
    Extract YYYY-MM-DD from a datetime representation.
    """

    value = normalize_datetime(value)

    return value[:10]


# ============================================================================
# THREAT PROBABILITY CONVERSION
# ============================================================================

def class1_to_threat_probability(class1_probability):
    """
    Convert class-1 probability into physical-threat probability.

    Dataset convention:

        class 0 = threat
        class 1 = non-threat

    Therefore:

        P(threat) = 1 - P(class 1)
    """

    class1_probability = np.asarray(
        class1_probability,
        dtype=np.float64,
    )

    threat_probability = (
        1.0 - class1_probability
    )

    return np.clip(
        threat_probability,
        0.0,
        1.0,
    )


# ============================================================================
# INITIAL THREAT DECISION
# ============================================================================

def threshold_threat_probability(
    threat_probability,
    threshold,
):
    """
    Convert threat probabilities into initial binary threat decisions.

    Parameters
    ----------
    threat_probability : array-like
        Probability of physical proximity/threat.

    threshold : float
        Threat decision threshold.

    Returns
    -------
    np.ndarray
        Boolean threat decisions.
    """

    threat_probability = np.asarray(
        threat_probability,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(threat_probability)
    ):
        raise ValueError(
            "Threat probabilities contain non-finite values."
        )

    if not (
        0.0 <= threshold <= 1.0
    ):
        raise ValueError(
            f"Threshold must be in [0,1], got {threshold}"
        )

    return (
        threat_probability >= threshold
    )


# ============================================================================
# TEMPORAL PERSISTENCE
# ============================================================================

def apply_temporal_persistence(
    threat_decisions,
    window=PERSISTENCE_WINDOW,
    required=PERSISTENCE_REQUIRED,
):
    """
    Apply temporal persistence to a chronological threat sequence.

    For every observation, inspect the current observation and the
    preceding observations within the persistence window.

    A final threat is emitted when at least `required` observations
    in the available window indicate threat.

    Example with window=3 and required=2:

        Raw:
            0 0 1 0 1 1

        Persistent:
            0 0 0 0 1 1

    The first few observations use the available history only.

    IMPORTANT:
        This function must be called separately for each day.
        It must never receive a sequence spanning multiple days.
    """

    decisions = np.asarray(
        threat_decisions,
        dtype=bool,
    )

    if decisions.ndim != 1:
        raise ValueError(
            "Threat decisions must be a 1D sequence."
        )

    if window <= 0:
        raise ValueError(
            "Persistence window must be positive."
        )

    if required <= 0:
        raise ValueError(
            "Required persistence count must be positive."
        )

    if required > window:
        raise ValueError(
            "Required persistence count cannot exceed "
            "the persistence window."
        )

    if len(decisions) == 0:
        return np.empty(
            0,
            dtype=bool,
        )

    persistent = np.zeros(
        len(decisions),
        dtype=bool,
    )

    for i in range(len(decisions)):

        start = max(
            0,
            i - window + 1,
        )

        local_window = decisions[
            start:i + 1
        ]

        threat_count = np.sum(
            local_window
        )

        persistent[i] = (
            threat_count >= required
        )

    return persistent


# ============================================================================
# DAY-WISE TEMPORAL PERSISTENCE
# ============================================================================

def apply_daywise_temporal_persistence(
    threat_decisions,
    datetimes,
    window=PERSISTENCE_WINDOW,
    required=PERSISTENCE_REQUIRED,
):
    """
    Apply temporal persistence independently within each day.

    This prevents temporal state from leaking across day boundaries.
    """

    threat_decisions = np.asarray(
        threat_decisions,
        dtype=bool,
    )

    datetimes = np.asarray(
        datetimes,
    )

    if len(threat_decisions) != len(datetimes):
        raise ValueError(
            "Threat decisions and datetimes are misaligned."
        )

    result = np.zeros(
        len(threat_decisions),
        dtype=bool,
    )

    day_strings = np.array(
        [
            get_day_string(dt)
            for dt in datetimes
        ]
    )

    unique_days = np.unique(
        day_strings
    )

    for day in unique_days:

        indices = np.flatnonzero(
            day_strings == day
        )

        if len(indices) == 0:
            continue

        # The cached representation should already be chronological.
        day_decisions = threat_decisions[
            indices
        ]

        result[
            indices
        ] = apply_temporal_persistence(
            day_decisions,
            window=window,
            required=required,
        )

    return result


# ============================================================================
# EVENT EXTRACTION
# ============================================================================

def extract_persistent_events(
    persistent_decisions,
    datetimes,
):
    """
    Extract contiguous persistent-threat events.

    Each event is represented by:

        start_index
        end_index
        duration_samples
        start_datetime
        end_datetime

    Events are never allowed to cross a day boundary.
    """

    persistent_decisions = np.asarray(
        persistent_decisions,
        dtype=bool,
    )

    datetimes = np.asarray(
        datetimes,
    )

    if len(persistent_decisions) != len(datetimes):
        raise ValueError(
            "Persistent decisions and datetimes are misaligned."
        )

    events = []

    day_strings = np.array(
        [
            get_day_string(dt)
            for dt in datetimes
        ]
    )

    start = None

    for i in range(
        len(persistent_decisions)
    ):

        if not persistent_decisions[i]:

            if start is not None:

                end = i - 1

                events.append(
                    {
                        "start_index": int(start),
                        "end_index": int(end),
                        "duration_samples": int(
                            end - start + 1
                        ),
                        "start_datetime": normalize_datetime(
                            datetimes[start]
                        ),
                        "end_datetime": normalize_datetime(
                            datetimes[end]
                        ),
                    }
                )

                start = None

            continue

        # Start a new event.
        if start is None:
            start = i
            continue

        # Never allow an event to cross a day boundary.
        if day_strings[i] != day_strings[start]:

            end = i - 1

            events.append(
                {
                    "start_index": int(start),
                    "end_index": int(end),
                    "duration_samples": int(
                        end - start + 1
                    ),
                    "start_datetime": normalize_datetime(
                        datetimes[start]
                    ),
                    "end_datetime": normalize_datetime(
                        datetimes[end]
                    ),
                }
            )

            start = i

    # Close final event.
    if start is not None:

        end = len(
            persistent_decisions
        ) - 1

        events.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "duration_samples": int(
                    end - start + 1
                ),
                "start_datetime": normalize_datetime(
                    datetimes[start]
                ),
                "end_datetime": normalize_datetime(
                    datetimes[end]
                ),
            }
        )

    return events


# ============================================================================
# SANITY CHECK
# ============================================================================

def sanity_check():
    """
    Validate H7 using:

        1. Real H1-H4 cached data.
        2. Synthetic classifier probabilities.
        3. Class-1 -> threat probability conversion.
        4. Threat thresholding.
        5. Day-wise temporal persistence.
        6. Persistent event extraction.

    No full model training is performed.
    """

    print()
    print("=" * 70)
    print("H7 TEMPORAL PERSISTENCE SANITY CHECK")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load frozen representation
    # ------------------------------------------------------------------------

    X, y, dt = load_cached_data()

    print()
    print("Frozen H1-H4 representation:")
    print(f"  Features:  {X.shape}")
    print(f"  Targets:   {y.shape}")
    print(f"  Datetimes: {dt.shape}")

    # ------------------------------------------------------------------------
    # Verify frozen representation
    # ------------------------------------------------------------------------

    assert X.shape[1] == EXPECTED_FEATURES
    assert len(X) == len(y)
    assert len(X) == len(dt)

    print()
    print("Frozen representation checks: PASS")

    # ------------------------------------------------------------------------
    # Synthetic probability sequence
    # ------------------------------------------------------------------------
    #
    # We construct a deliberately mixed sequence:
    #
    #       high high low high low high high ...
    #
    # This allows us to verify that isolated probability spikes
    # are suppressed while persistent threat sequences survive.
    #

    synthetic_threat_probability = np.array(
        [
            0.10,
            0.85,
            0.90,
            0.15,
            0.88,
            0.20,
            0.92,
            0.91,
            0.90,
            0.10,
        ],
        dtype=np.float64,
    )

    synthetic_class1_probability = (
        1.0
        - synthetic_threat_probability
    )

    # ------------------------------------------------------------------------
    # Probability conversion
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
            "Probability conversion failed."
        )

    print("Probability conversion: PASS")

    # ------------------------------------------------------------------------
    # Initial threshold
    # ------------------------------------------------------------------------

    threshold = 0.50

    raw_decisions = (
        threshold_threat_probability(
            recovered_threat_probability,
            threshold,
        )
    )

    expected_raw = np.array(
        [
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
        ],
        dtype=bool,
    )

    if not np.array_equal(
        raw_decisions,
        expected_raw,
    ):
        raise AssertionError(
            "Raw threshold decisions are incorrect."
        )

    print("Threat thresholding: PASS")

    # ------------------------------------------------------------------------
    # Synthetic datetimes
    # ------------------------------------------------------------------------
    #
    # The sequence deliberately contains a day boundary.
    #
    # The final two observations of day 1 are threats.
    # The first observation of day 2 is a threat.
    #
    # H7 must NOT use day-1 history to trigger day-2 persistence.
    #

    synthetic_dt = np.array(
        [
            "2023-06-16T00:00:00",
            "2023-06-16T00:00:10",
            "2023-06-16T00:00:20",
            "2023-06-16T00:00:30",
            "2023-06-16T00:00:40",
            "2023-06-16T00:00:50",
            "2023-06-16T00:01:00",
            "2023-06-16T00:01:10",
            "2023-06-17T00:00:00",
            "2023-06-17T00:00:10",
        ],
        dtype=str,
    )

    # ------------------------------------------------------------------------
    # Day-wise persistence
    # ------------------------------------------------------------------------

    persistent_decisions = (
        apply_daywise_temporal_persistence(
            raw_decisions,
            synthetic_dt,
            window=PERSISTENCE_WINDOW,
            required=PERSISTENCE_REQUIRED,
        )
    )

    # For 2-of-3 persistence:
    #
    # raw:
    # F T T F T F T T T F
    #
    # expected:
    # F F T T T T T T F F
    #
    # The first observation of the second day must remain false because
    # its previous history from day 1 cannot be used.

    expected_persistent = np.array(
        [
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
        ],
        dtype=bool,
    )

    if not np.array_equal(
        persistent_decisions,
        expected_persistent,
    ):
        raise AssertionError(
            "Temporal persistence output is incorrect.\n"
            f"Expected: {expected_persistent}\n"
            f"Got:      {persistent_decisions}"
        )

    print("Day-wise temporal persistence: PASS")

    # ------------------------------------------------------------------------
    # Boundary leakage check
    # ------------------------------------------------------------------------

    if persistent_decisions[8]:
        raise AssertionError(
            "Temporal persistence leaked across the day boundary."
        )

    print("Day-boundary isolation: PASS")

    # ------------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------------

    events = extract_persistent_events(
        persistent_decisions,
        synthetic_dt,
    )

    if len(events) == 0:
        raise AssertionError(
            "Expected at least one persistent event."
        )

    # Verify that no event crosses days.
    for event in events:

        start_day = get_day_string(
            event["start_datetime"]
        )

        end_day = get_day_string(
            event["end_datetime"]
        )

        if start_day != end_day:
            raise AssertionError(
                "Extracted event crosses a day boundary."
            )

    print("Persistent event extraction: PASS")

    # ------------------------------------------------------------------------
    # Configuration checks
    # ------------------------------------------------------------------------

    if PERSISTENCE_WINDOW != 3:
        raise AssertionError(
            "Unexpected persistence window."
        )

    if PERSISTENCE_REQUIRED != 2:
        raise AssertionError(
            "Unexpected persistence requirement."
        )

    if PERSISTENCE_REQUIRED > PERSISTENCE_WINDOW:
        raise AssertionError(
            "Invalid persistence configuration."
        )

    print("H7 configuration checks: PASS")

    # ------------------------------------------------------------------------
    # Display result
    # ------------------------------------------------------------------------

    print()
    print("H7 configuration:")
    print(
        f"  Persistence window: "
        f"{PERSISTENCE_WINDOW} observations"
    )
    print(
        f"  Required threats: "
        f"{PERSISTENCE_REQUIRED}"
    )
    print(
        f"  Initial threat threshold: "
        f"{threshold:.2f}"
    )

    print()
    print("Synthetic sequence:")
    print(
        "  Raw decisions:        "
        f"{raw_decisions.astype(int)}"
    )
    print(
        "  Persistent decisions: "
        f"{persistent_decisions.astype(int)}"
    )

    print()
    print(
        f"Persistent events detected: "
        f"{len(events)}"
    )

    for i, event in enumerate(
        events,
        start=1,
    ):
        print(
            f"  Event {i}: "
            f"{event['start_datetime']} -> "
            f"{event['end_datetime']} "
            f"({event['duration_samples']} samples)"
        )

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("H7 SANITY CHECK: PASS")
    print("=" * 70)
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point.
    """

    if (
        len(sys.argv) != 2
        or sys.argv[1] != "--sanity"
    ):
        print(
            "Usage:\n"
            "  python scripts/run_h7_temporal_persistence.py "
            "--sanity"
        )

        print()

        print(
            "Full H7 evaluation is intentionally deferred until "
            "the cumulative H5 + H6 + H7 experiment."
        )

        sys.exit(1)

    sanity_check()


if __name__ == "__main__":
    main()
