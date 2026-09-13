import os
import sys
import numpy as np
import pandas as pd
import h5py

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, REPO_ROOT)

H5_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "dataset_sensor_range_1440_1690_0.h5",
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

THREAT_THRESHOLD = 1000.0

N_SECONDS = 50
N_OVERLAPPING_SECONDS = -10
SAMPLE_SECONDS = 10

RANDOM_STATE = 42

EXPECTED_BASE_FEATURES = 2800
EXPECTED_H5_FEATURES = 2801

# H3 = 25 features/frame × 5 frames = 125
# H4 = 25 H3 features × 6 temporal descriptors = 150
# Together = 275 compact distance-prediction inputs.
#
# H1 = 500/frame × 5 = 2500
# H2 = 5/frame × 5 = 25
# H3 = 25/frame × 5 = 125
# H4 = 150
# Total H1-H4 = 2800
#
# For the auxiliary distance model we use only the compact
# H3 + H4 descriptors. This avoids expensive 2800-D regression
# while retaining the spatial/coherence and temporal information
# most relevant to continuous vessel distance.


# ============================================================
# MODELS
# ============================================================

def make_distance_model():
    """
    Lightweight auxiliary distance model.

    Ridge regression is used only to generate a predicted
    continuous-distance feature. Ground-truth distance is never
    passed to the binary classifier.
    """

    return Ridge(
        alpha=100.0,
        solver="lsqr",
        max_iter=1000,
    )


def make_classifier(seed=42):
    """
    Same XGBoost classifier family used in the baseline/H1-H4
    experiments, with one additional H5 feature.
    """

    return XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        learning_rate=0.05,
        max_depth=10,
        n_estimators=500,
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
    )


# ============================================================
# LOAD CACHED H1-H4 FEATURES
# ============================================================

def load_cache():

    print("Loading cached H1-H4 representation...")

    X_features = np.load(
        FEATURE_PATH,
        mmap_mode="r",
    )

    y_binary = np.load(
        TARGET_PATH
    )

    dt_reduced = np.load(
        DATETIME_PATH
    )

    print("Cached features:", X_features.shape)
    print("Binary targets:", y_binary.shape)
    print("Datetimes:", dt_reduced.shape)

    if X_features.shape != (
        74767,
        EXPECTED_BASE_FEATURES,
    ):
        raise RuntimeError(
            "Unexpected cached feature shape: "
            f"{X_features.shape}"
        )

    if y_binary.shape[0] != X_features.shape[0]:
        raise RuntimeError(
            "Target/feature length mismatch."
        )

    if dt_reduced.shape[0] != X_features.shape[0]:
        raise RuntimeError(
            "Datetime/feature length mismatch."
        )

    print("Cache dimensionality/alignment: PASS")

    return (
        X_features,
        y_binary,
        dt_reduced,
    )


# ============================================================
# LOAD RAW CONTINUOUS DISTANCE
# ============================================================

def load_raw_distance():

    print("\nLoading raw continuous distance...")

    with h5py.File(
        H5_PATH,
        "r",
    ) as f:

        y_distance = np.array(
            f["y"],
            dtype=np.float32,
        )

        datetimes = np.array(
            [
                dt.decode("utf-8")
                for dt in f["datetimes"]
            ]
        )

    print(
        "Raw distance:",
        y_distance.shape,
    )

    print(
        "Raw datetimes:",
        datetimes.shape,
    )

    return (
        y_distance,
        datetimes,
    )


# ============================================================
# BUILD CONTINUOUS DISTANCE TARGETS
# ============================================================

def build_distance_targets(
    y_distance,
    datetimes,
):

    print(
        "\nBuilding aligned continuous "
        "distance targets..."
    )

    n_samples_per_group = (
        N_SECONDS // SAMPLE_SECONDS
    )

    if N_OVERLAPPING_SECONDS < 0:

        n_overlap_samples = (
            n_samples_per_group
            + (
                N_OVERLAPPING_SECONDS
                // SAMPLE_SECONDS
            )
        )

    else:

        n_overlap_samples = (
            N_OVERLAPPING_SECONDS
            // SAMPLE_SECONDS
        )

    step = (
        n_samples_per_group
        - n_overlap_samples
    )

    if step <= 0:
        raise ValueError(
            "Invalid overlap configuration."
        )

    reduced_distance = []
    reduced_dt = []

    i = 0

    while i < len(y_distance):

        group_distance = y_distance[
            i:i + n_samples_per_group
        ]

        group_dt = datetimes[
            i:i + n_samples_per_group
        ]

        if len(group_distance) < n_samples_per_group:
            break

        # Same central-frame target used by the
        # regression formulation.
        mid_index = (
            len(group_distance) // 2
        )

        reduced_distance.append(
            float(
                group_distance[mid_index]
            )
        )

        # Same timestamp convention used by
        # TripletReducer.
        reduced_dt.append(
            min(group_dt)
        )

        i += step

    reduced_distance = np.asarray(
        reduced_distance,
        dtype=np.float32,
    )

    reduced_dt = np.asarray(
        reduced_dt
    )

    print(
        "Reduced distance:",
        reduced_distance.shape,
    )

    print(
        "Reduced datetime:",
        reduced_dt.shape,
    )

    if len(reduced_distance) != 74767:

        raise RuntimeError(
            "Unexpected reduced distance length: "
            f"{len(reduced_distance)}"
        )

    return (
        reduced_distance,
        reduced_dt,
    )


# ============================================================
# ALIGNMENT CHECK
# ============================================================

def verify_alignment(
    cache_dt,
    distance_dt,
):

    if not np.array_equal(
        cache_dt,
        distance_dt,
    ):

        mismatch = np.where(
            cache_dt != distance_dt
        )[0]

        raise RuntimeError(
            "Datetime alignment failure. "
            f"First mismatch: {mismatch[0]}"
        )

    print(
        "Feature/distance datetime "
        "alignment: PASS"
    )


# ============================================================
# EXTRACT COMPACT AUXILIARY FEATURES
# ============================================================

def extract_auxiliary_features(
    X_features
):
    """
    Extract the compact H3 + H4 portion of the
    cached H1-H4 representation.

    Layout per 5-frame instance:

        H1 = 2500
        H2 =   25
        H3 =  125
        H4 =  150
        ----------------
        total = 2800

    We use:

        H3 + H4 = 275 features

    for the auxiliary continuous-distance model.

    This substantially reduces the cost of the
    distance-learning stage while retaining the
    spatial coherence and temporal dynamics
    introduced by H3/H4.
    """

    h1_h2_features = 2525

    aux_features = np.asarray(
        X_features[
            :,
            h1_h2_features:
        ],
        dtype=np.float32,
    )

    if aux_features.shape != (
        74767,
        275,
    ):

        raise RuntimeError(
            "Unexpected auxiliary feature shape: "
            f"{aux_features.shape}"
        )

    print(
        "\nAuxiliary distance features:",
        aux_features.shape,
    )

    print(
        "Auxiliary dimensionality: PASS"
    )

    return aux_features


# ============================================================
# STRICT OUTER-FOLD CROSS-FITTING
# ============================================================

def cross_fitted_distance_predictions(
    X_aux,
    y_distance,
    days,
    train_days,
    test_day,
):

    print(
        "\nDistance cross-fitting for "
        f"outer test day: {test_day}"
    )

    train_days = list(train_days)

    # Deterministic grouped cross-fitting.
    group_a_days = train_days[:5]
    group_b_days = train_days[5:]

    mask_a = np.isin(
        days,
        group_a_days,
    )

    mask_b = np.isin(
        days,
        group_b_days,
    )

    mask_test = (
        days == test_day
    )

    print(
        f"  Group A: {len(group_a_days)} days, "
        f"{mask_a.sum()} samples"
    )

    print(
        f"  Group B: {len(group_b_days)} days, "
        f"{mask_b.sum()} samples"
    )

    # --------------------------------------------------------
    # A -> B
    # --------------------------------------------------------

    print(
        "  Training lightweight distance "
        "model A -> B..."
    )

    model_ab = make_distance_model()

    X_a = np.asarray(
        X_aux[mask_a],
        dtype=np.float32,
    )

    X_b = np.asarray(
        X_aux[mask_b],
        dtype=np.float32,
    )

    y_a = y_distance[mask_a]

    model_ab.fit(
        X_a,
        y_a,
    )

    pred_b = model_ab.predict(
        X_b
    ).astype(np.float32)

    del model_ab
    del X_a
    del X_b
    del y_a

    # --------------------------------------------------------
    # B -> A
    # --------------------------------------------------------

    print(
        "  Training lightweight distance "
        "model B -> A..."
    )

    model_ba = make_distance_model()

    X_b = np.asarray(
        X_aux[mask_b],
        dtype=np.float32,
    )

    X_a = np.asarray(
        X_aux[mask_a],
        dtype=np.float32,
    )

    y_b = y_distance[mask_b]

    model_ba.fit(
        X_b,
        y_b,
    )

    pred_a = model_ba.predict(
        X_a
    ).astype(np.float32)

    del model_ba
    del X_a
    del X_b
    del y_b

    # --------------------------------------------------------
    # Assemble OOF predictions
    # --------------------------------------------------------

    oof_predictions = np.empty(
        X_aux.shape[0],
        dtype=np.float32,
    )

    oof_predictions[mask_a] = pred_a
    oof_predictions[mask_b] = pred_b

    del pred_a
    del pred_b

    # --------------------------------------------------------
    # Final distance model
    # Train only on the nine outer training days.
    # Then predict the completely unseen test day.
    # --------------------------------------------------------

    print(
        "  Training final lightweight "
        "distance model on 9 training days..."
    )

    model_final = make_distance_model()

    X_train_distance = np.asarray(
        X_aux[~mask_test],
        dtype=np.float32,
    )

    y_train_distance = y_distance[
        ~mask_test
    ]

    X_test_distance = np.asarray(
        X_aux[mask_test],
        dtype=np.float32,
    )

    model_final.fit(
        X_train_distance,
        y_train_distance,
    )

    pred_test = model_final.predict(
        X_test_distance
    ).astype(np.float32)

    del model_final
    del X_train_distance
    del y_train_distance
    del X_test_distance

    return (
        oof_predictions,
        pred_test,
    )


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():

    (
        X_features,
        y_binary,
        cache_dt,
    ) = load_cache()

    (
        y_distance_raw,
        raw_dt,
    ) = load_raw_distance()

    (
        y_distance,
        distance_dt,
    ) = build_distance_targets(
        y_distance_raw,
        raw_dt,
    )

    del y_distance_raw
    del raw_dt

    verify_alignment(
        cache_dt,
        distance_dt,
    )

    del distance_dt

    days = np.array(
        [
            str(dt)[:10]
            for dt in cache_dt
        ]
    )

    all_days = sorted(
        np.unique(days)
    )

    print("\nUnique days:")

    for day in all_days:

        print(
            f"  {day}: "
            f"{(days == day).sum()} samples"
        )

    # --------------------------------------------------------
    # Extract compact H3 + H4 auxiliary representation
    # --------------------------------------------------------

    X_aux = extract_auxiliary_features(
        X_features
    )

    # --------------------------------------------------------
    # Outer 10-fold evaluation
    # --------------------------------------------------------

    fold_results = []

    for fold_idx, test_day in enumerate(
        all_days,
        start=1,
    ):

        train_days = [
            d
            for d in all_days
            if d != test_day
        ]

        train_mask = (
            days != test_day
        )

        test_mask = (
            days == test_day
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"FOLD {fold_idx}/10 — "
            f"TEST DAY {test_day}"
        )

        print(
            "=" * 70
        )

        (
            oof_distance,
            test_distance,
        ) = cross_fitted_distance_predictions(
            X_aux,
            y_distance,
            days,
            train_days,
            test_day,
        )

        # ----------------------------------------------------
        # Append predicted distance as H5 feature
        # ----------------------------------------------------

        X_train_base = np.asarray(
            X_features[train_mask],
            dtype=np.float32,
        )

        X_test_base = np.asarray(
            X_features[test_mask],
            dtype=np.float32,
        )

        X_train = np.column_stack(
            [
                X_train_base,
                oof_distance[train_mask],
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        X_test = np.column_stack(
            [
                X_test_base,
                test_distance,
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        del X_train_base
        del X_test_base
        del oof_distance
        del test_distance

        if X_train.shape[1] != (
            EXPECTED_H5_FEATURES
        ):

            raise RuntimeError(
                "Unexpected H5 training dimension: "
                f"{X_train.shape[1]}"
            )

        if X_test.shape[1] != (
            EXPECTED_H5_FEATURES
        ):

            raise RuntimeError(
                "Unexpected H5 test dimension: "
                f"{X_test.shape[1]}"
            )

        print(
            "Classifier training shape:",
            X_train.shape,
        )

        print(
            "Classifier test shape:",
            X_test.shape,
        )

        print(
            "H5 dimensionality: PASS"
        )

        # ----------------------------------------------------
        # Binary classifier
        # ----------------------------------------------------

        model = make_classifier(
            RANDOM_STATE
        )

        model.fit(
            X_train,
            y_binary[train_mask],
        )

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        y_true = y_binary[
            test_mask
        ]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        f1 = f1_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
        )

        recall = recall_score(
            y_true,
            y_pred,
        )

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        print(
            f"Accuracy:  {accuracy:.4f}"
        )

        print(
            f"F1:        {f1:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"AUC:       {auc:.4f}"
        )

        print(
            "Confusion matrix:"
        )

        print(cm)

        fold_results.append(
            {
                "fold": fold_idx,
                "test_day": test_day,
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "tn": cm[0, 0],
                "fp": cm[0, 1],
                "fn": cm[1, 0],
                "tp": cm[1, 1],
                "support": len(y_pred),
            }
        )

        del model
        del X_train
        del X_test
        del y_pred
        del y_prob
        del y_true

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = pd.DataFrame(
        fold_results
    )

    output_dir = os.path.join(
        REPO_ROOT,
        "results",
        "DAS-XGBoost-H5",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    results_path = os.path.join(
        output_dir,
        "h5_fold_metrics.csv",
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print(
        "\nResults saved:"
    )

    print(
        results_path
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "H5 EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        results.to_string(
            index=False
        )
    )

    # ========================================================
    # POOLED GLOBAL METRICS
    # ========================================================

    total = results[
        "support"
    ].sum()

    total_tn = results[
        "tn"
    ].sum()

    total_fp = results[
        "fp"
    ].sum()

    total_fn = results[
        "fn"
    ].sum()

    total_tp = results[
        "tp"
    ].sum()

    global_accuracy = (
        total_tn + total_tp
    ) / total

    global_precision = (
        total_tp
        / (
            total_tp
            + total_fp
        )
    )

    global_recall = (
        total_tp
        / (
            total_tp
            + total_fn
        )
    )

    global_f1 = (
        2
        * global_precision
        * global_recall
        / (
            global_precision
            + global_recall
        )
    )

    summary = pd.DataFrame(
        [
            {
                "accuracy": global_accuracy,
                "f1": global_f1,
                "precision": global_precision,
                "recall": global_recall,
                "auc_mean_fold": results[
                    "auc"
                ].mean(),
                "total_tn": total_tn,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "total_tp": total_tp,
                "support": total,
            }
        ]
    )

    summary_path = os.path.join(
        output_dir,
        "h5_summary.csv",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nGlobal summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSummary saved:"
    )

    print(
        summary_path
    )


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check():

    print(
        "H5 lightweight distance-aware "
        "sanity check"
    )

    print(
        "Expected H1-H4 features:",
        EXPECTED_BASE_FEATURES,
    )

    print(
        "Expected H5 features:",
        EXPECTED_H5_FEATURES,
    )

    print(
        "Auxiliary distance features:",
        "H3 + H4 = 275",
    )

    print(
        "Distance target:",
        "central frame of each 5-frame group",
    )

    print(
        "Cross-fitting:",
        "strict outer-fold two-way grouped "
        "cross-fitting",
    )

    print(
        "Distance model:",
        "Ridge regression",
    )

    (
        X_features,
        y_binary,
        cache_dt,
    ) = load_cache()

    (
        y_distance_raw,
        raw_dt,
    ) = load_raw_distance()

    (
        y_distance,
        distance_dt,
    ) = build_distance_targets(
        y_distance_raw,
        raw_dt,
    )

    verify_alignment(
        cache_dt,
        distance_dt,
    )

    X_aux = extract_auxiliary_features(
        X_features
    )

    if not np.isfinite(
        y_distance
    ).all():

        raise RuntimeError(
            "Non-finite distance target detected."
        )

    if not np.isfinite(
        X_aux
    ).all():

        raise RuntimeError(
            "Non-finite auxiliary feature detected."
        )

    print(
        "Distance target finite:",
        True,
    )

    print(
        "Auxiliary features finite:",
        True,
    )

    print(
        "H5 LIGHTWEIGHT SANITY CHECK: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if "--sanity" in sys.argv:

        sanity_check()

    else:

        run_experiment()
