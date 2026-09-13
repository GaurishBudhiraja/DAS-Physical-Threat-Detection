import sys
import os
import h5py
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data_splitter_hdf5 import TripletReducer

H5_PATH = "data/dataset_sensor_range_1440_1690_0.h5"

print("=" * 60)
print("H5 DATA / REPRESENTATION SANITY CHECK")
print("=" * 60)

# ---------------------------------------------------------
# Load a tiny subset
# ---------------------------------------------------------
with h5py.File(H5_PATH, "r") as f:
    X = np.array(f["X"][:50], dtype=np.float32)
    y_distance = np.array(f["y"][:50], dtype=np.float32)
    dt = np.array(
        [d.decode("utf-8") for d in f["datetimes"][:50]]
    )

print(f"Raw X shape:        {X.shape}")
print(f"Raw distance shape: {y_distance.shape}")
print(f"Datetime shape:     {dt.shape}")

assert X.shape == (50, 250, 100)
assert y_distance.shape == (50,)
assert dt.shape == (50,)

print("Raw data checks: PASS")


# ---------------------------------------------------------
# Create binary target exactly as H5 classification does
# ---------------------------------------------------------
threshold = 1000.0

y_binary = (y_distance > threshold).astype(np.int8)

print(f"Binary classes: {np.unique(y_binary)}")
print(
    f"Class counts: "
    f"{dict(zip(*np.unique(y_binary, return_counts=True)))}"
)

assert set(np.unique(y_binary)).issubset({0, 1})

print("Binary target check: PASS")


# ---------------------------------------------------------
# Build H1-H4 representation
# ---------------------------------------------------------
reducer = TripletReducer(
    X=X,
    y=y_binary,
    dt=dt,
    n_seconds=50,
    n_overlapping_seconds=-10,
    join_higher_classes=True,
    average_signals="spatial_spectral_coherence_features",
    apply_log=True,
    use_mid_target=True,
    sample_seconds=10,
)

X_h4, y_reduced, dt_reduced = reducer.reduce_triplets()

print()
print(f"H1-H4 representation shape: {X_h4.shape}")
print(f"Reduced binary target shape: {y_reduced.shape}")
print(f"Reduced datetime shape:      {len(dt_reduced)}")

assert X_h4.ndim == 2
assert X_h4.shape[1] == 2800
assert len(y_reduced) == X_h4.shape[0]
assert len(dt_reduced) == X_h4.shape[0]

print("H1-H4 dimensionality: PASS")


# ---------------------------------------------------------
# Build continuous distance target using same grouping
# ---------------------------------------------------------
n_samples_per_group = 5
step = 1

distance_targets = []
distance_dt = []

i = 0

while i + n_samples_per_group <= len(y_distance):

    group_distance = y_distance[
        i:i + n_samples_per_group
    ]

    group_dt = dt[
        i:i + n_samples_per_group
    ]

    # Central frame = same temporal target convention
    mid_index = len(group_distance) // 2

    distance_targets.append(
        group_distance[mid_index]
    )

    distance_dt.append(
        min(group_dt)
    )

    i += step

distance_targets = np.asarray(
    distance_targets,
    dtype=np.float32
)

distance_dt = np.asarray(distance_dt)

print()
print(
    f"Distance target shape:       "
    f"{distance_targets.shape}"
)
print(
    f"Distance datetime shape:     "
    f"{distance_dt.shape}"
)

assert len(distance_targets) == X_h4.shape[0]
assert len(distance_dt) == X_h4.shape[0]

# Critical alignment check
assert np.array_equal(
    np.asarray(dt_reduced),
    distance_dt
)

print("Distance/feature alignment: PASS")


# ---------------------------------------------------------
# H5 augmentation simulation
# ---------------------------------------------------------
# Simulate predicted distance with the correct number of
# samples. This checks the final feature interface.
predicted_distance = np.zeros(
    X_h4.shape[0],
    dtype=np.float32
)

X_h5 = np.column_stack([
    X_h4,
    predicted_distance
])

print()
print(f"H5 representation shape: {X_h5.shape}")

assert X_h5.shape[1] == 2801
assert X_h5.shape[0] == X_h4.shape[0]

print("H5 dimensionality: PASS")


# ---------------------------------------------------------
# Finite-value checks
# ---------------------------------------------------------
assert np.isfinite(X_h4).all()
assert np.isfinite(distance_targets).all()
assert np.isfinite(X_h5).all()

print("Finite-value check: PASS")


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print()
print("=" * 60)
print("H5 DATA SANITY CHECK: PASS")
print("=" * 60)
print("H1-H4 features : 2800")
print("H5 features    : 2801")
print("Distance target: central frame")
print("Alignment      : verified")
print("NaN / Inf      : none")
print("=" * 60)
