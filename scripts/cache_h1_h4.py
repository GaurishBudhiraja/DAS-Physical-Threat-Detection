import os
import sys
import numpy as np
import h5py

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, REPO_ROOT)

from src.data_splitter_hdf5 import TripletReducer


H5_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "dataset_sensor_range_1440_1690_0.h5"
)

CACHE_DIR = os.path.join(
    REPO_ROOT,
    "research",
    "cache"
)

os.makedirs(CACHE_DIR, exist_ok=True)

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


N_SECONDS = 50
N_OVERLAPPING_SECONDS = -10
SAMPLE_SECONDS = 10


def main():

    print("Loading HDF5 dataset...")

    with h5py.File(H5_PATH, "r") as f:

        X = np.array(
            f["X"],
            dtype=np.float32
        )

        y_distance = np.array(
            f["y"],
            dtype=np.float32
        )

        datetimes = np.array([
            dt.decode("utf-8")
            for dt in f["datetimes"]
        ])

    print("Raw X:", X.shape)
    print("Raw distance:", y_distance.shape)
    print("Raw datetime:", datetimes.shape)

    y_binary = (
        y_distance > 1000.0
    ).astype(np.int8)

    print("\nBuilding exact H1-H4 representation...")
    print("This is the ONE expensive preprocessing run.")

    reducer = TripletReducer(
        X,
        y_binary,
        datetimes,
        ships=None,
        n_seconds=N_SECONDS,
        n_overlapping_seconds=N_OVERLAPPING_SECONDS,
        join_higher_classes=True,
        average_signals="spatial_spectral_coherence_features",
        apply_log=True,
        epsilon=1e-23,
        time_offset_seconds=None,
        use_mid_target=True,
        sample_seconds=SAMPLE_SECONDS,
        center_truth=False,
    )

    X_reduced, y_reduced, dt_reduced = (
        reducer.reduce_triplets()
    )

    del reducer
    del X

    X_reduced = np.asarray(
        X_reduced,
        dtype=np.float32
    )

    y_reduced = np.asarray(
        y_reduced,
        dtype=np.int8
    )

    dt_reduced = np.asarray(
        dt_reduced
    )

    print("\nFeature extraction complete.")

    print("H1-H4 features:", X_reduced.shape)
    print("Targets:", y_reduced.shape)
    print("Datetimes:", dt_reduced.shape)

    if X_reduced.shape != (74767, 2800):
        raise RuntimeError(
            f"Unexpected feature shape: "
            f"{X_reduced.shape}; expected (74767, 2800)"
        )

    if y_reduced.shape != (74767,):
        raise RuntimeError(
            f"Unexpected target shape: "
            f"{y_reduced.shape}; expected (74767,)"
        )

    if dt_reduced.shape != (74767,):
        raise RuntimeError(
            f"Unexpected datetime shape: "
            f"{dt_reduced.shape}; expected (74767,)"
        )

    print("Dimension validation: PASS")

    np.save(
        FEATURE_PATH,
        X_reduced
    )

    np.save(
        TARGET_PATH,
        y_reduced
    )

    np.save(
        DATETIME_PATH,
        dt_reduced
    )

    print("\nCache saved:")

    print(FEATURE_PATH)
    print(TARGET_PATH)
    print(DATETIME_PATH)

    print("\nCache validation:")
    print(
        "Features:",
        np.load(FEATURE_PATH, mmap_mode="r").shape
    )
    print(
        "Targets:",
        np.load(TARGET_PATH, mmap_mode="r").shape
    )
    print(
        "Datetimes:",
        np.load(DATETIME_PATH, mmap_mode="r").shape
    )

    print("\nH1-H4 CACHE CREATION: PASS")


if __name__ == "__main__":
    main()
