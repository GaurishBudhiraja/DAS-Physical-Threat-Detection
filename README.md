# DAS Physical Threat Detection for Submarine Optical-Fiber Cables

### A Spatiotemporal Machine-Learning Framework for Physical Vessel-Threat Detection Using Distributed Acoustic Sensing

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![DAS](https://img.shields.io/badge/Sensing-DAS-6f42c1) ![Machine
Learning](https://img.shields.io/badge/Domain-Machine%20Learning-blue)
![Research](https://img.shields.io/badge/Research-IEEE%20Style-red)
![Status](https://img.shields.io/badge/Status-Research%20Complete-success)
![License](https://img.shields.io/badge/License-GPLv3-blue)

</p>

------------------------------------------------------------------------

## Author

**Gaurish Budhiraja**  
B.Tech Computer Science and Engineering  
Vellore Institute of Technology, Chennai, India

**Research Area:** Distributed Acoustic Sensing · Submarine Cable
Protection · Machine Learning · Physical Threat Detection ·
Spatiotemporal Signal Analysis

**GitHub:** https://github.com/GaurishBudhiraja

------------------------------------------------------------------------

# Overview

Submarine optical-fiber cables form a critical component of global
communications infrastructure. Although these cables are primarily
designed for data transmission, the optical fiber can also function as a
distributed sensing medium.

**Distributed Acoustic Sensing (DAS)** transforms an optical fiber into
a dense array of virtual acoustic sensing locations. Mechanical
disturbances occurring near the cable generate measurable changes in the
optical signal, allowing the fiber to continuously sense physical
activity along its route.

This repository presents a **spatiotemporal machine-learning framework
for physical vessel-threat detection around submarine optical-fiber
cables using Distributed Acoustic Sensing**.

The research investigates whether the spatial, spectral, morphological,
coherence-based, and temporal structure contained within DAS
measurements can improve detection of vessels operating within a defined
proximity to a submarine cable.

The research is organized into **nine experimental hypotheses (H1–H9)**
and culminates in an integrated physical-threat detection pipeline
combining:

- Spatial feature engineering
- Spectral characterization
- Spatial morphology
- Adjacent-channel coherence
- Temporal dynamics
- Higher-order temporal dynamics
- Fold-contained feature selection
- Distance-aware auxiliary learning
- Probability calibration
- Threat-aware decision making
- Temporal persistence

The final system is evaluated using **10-fold leave-one-day-out
cross-validation**, with each complete recording day held out as an
unseen test fold.

------------------------------------------------------------------------

# Research Objective

The primary research question is:

> **Can engineered spatiotemporal representations of Distributed
> Acoustic Sensing measurements improve the detection of physical vessel
> threats around submarine optical-fiber cables compared with a
> conventional channel-averaged XGBoost baseline?**

The research focuses exclusively on the **physical sensing layer**.

This project does **not** attempt to detect network-layer attacks such
as:

- Man-in-the-middle attacks
- Packet injection
- Network sniffing
- Protocol attacks
- Network traffic manipulation

The target phenomenon is a **physical vessel presence/proximity event
detected through DAS measurements**.

------------------------------------------------------------------------

# System Architecture

The final research system follows a staged architecture:

``` text
                         SUBMARINE DAS DATA
                                  │
                                  ▼
                       Original DAS Preprocessing
                                  │
                                  ▼
                        5-Frame Temporal Window
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ H1 — Spatial Features    │
                    │ H2 — Spectral Features   │
                    │ H3 — Spatial Coherence   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                       H4 — Temporal Dynamics
                                  │
                                  ▼
                    H9 — Higher-Order Dynamics
                                  │
                                  ▼
                       H8 — Feature Selection
                            Top-1000 Features
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             XGBoost Features          H5 Auxiliary
                                     Distance Branch
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         XGBoost Classifier
                                  │
                                  ▼
                     H6 Probability Calibration
                                  │
                                  ▼
                    Threat-Aware Decision Threshold
                                  │
                                  ▼
                       H7 Temporal Persistence
                                  │
                                  ▼
                       PHYSICAL THREAT ALERT
```

Final pipeline:

``` text
H1 + H2 + H3 + H4 + H9
            ↓
     H8 Feature Selection
            ↓
       H5 Auxiliary
    Distance Prediction
            ↓
       XGBoost Classifier
            ↓
      H6 Calibration
            ↓
    Threat-Aware Threshold
            ↓
      H7 Persistence
            ↓
    Physical Threat Alert
```

------------------------------------------------------------------------

# Dataset

The experiments use the publicly released **Marlinks-NS Distributed
Acoustic Sensing dataset** for vessel detection and distance estimation
in submarine optical-fiber cables.

## Dataset Characteristics

| Property                      |               Value |
|-------------------------------|--------------------:|
| Total raw observations        |          **74,771** |
| Final evaluation observations |          **74,767** |
| Spatial channels              |             **250** |
| Frequency bands               |             **100** |
| Source feature shape          |       **250 × 100** |
| Frequency range               |         **4–98 Hz** |
| Observation period            | **16–25 June 2023** |
| Cable segment                 |         **2,553 m** |
| Source temporal window        |      **10 seconds** |
| Evaluation days               |              **10** |

Each source observation is represented as:

``` text
250 spatial channels × 100 frequency bands
```

The dataset contains:

- DAS energy-band measurements
- UTC timestamps
- Vessel-distance information
- Vessel metadata

Three noisy spatial channels are excluded during processing:

``` text
Channel 59
Channel 60
Channel 61
```

The frequency representation consists of 100 logarithmically spaced
energy bands, excluding the 49–51 Hz region.

------------------------------------------------------------------------

# Physical Threat Definition

The physical-threat classification task uses a **1,000 m proximity
threshold**.

``` text
Class 0 → PHYSICAL THREAT
           Vessel distance ≤ 1000 m

Class 1 → NON-THREAT
           Vessel distance > 1000 m
```

Because the threat class is **Class 0**, the primary operational metrics
are reported from the threat perspective:

- Threat Precision
- Threat Recall
- Threat F1
- Missed Threats
- False Proximity Alerts

------------------------------------------------------------------------

# Experimental Protocol

## 10-Fold Leave-One-Day-Out Evaluation

DAS observations are temporally correlated. Randomly splitting
individual observations can introduce temporal leakage by placing highly
related measurements into both training and test sets.

This research uses:

> **10-fold leave-one-day-out cross-validation**

Each complete recording day is treated as one independent test fold.

``` text
Fold 1  → 2023-06-16
Fold 2  → 2023-06-17
Fold 3  → 2023-06-18
Fold 4  → 2023-06-19
Fold 5  → 2023-06-20
Fold 6  → 2023-06-21
Fold 7  → 2023-06-22
Fold 8  → 2023-06-23
Fold 9  → 2023-06-24
Fold 10 → 2023-06-25
```

For every fold:

``` text
9 complete days → Training / model development
1 complete day  → Completely unseen testing
```

Final evaluation population:

``` text
74,767 observations
```

------------------------------------------------------------------------

# Machine-Learning Model

The primary classification model is **XGBoost**.

## XGBoost Configuration

``` text
Objective       : binary:logistic
Booster         : gbtree
Learning Rate   : 0.05
Maximum Depth   : 10
Estimators      : 500
Random Seed     : 42
Tree Method     : hist
Parallel Jobs   : 8
```

XGBoost is used as the primary nonlinear classifier for the engineered
DAS representation and for fold-contained feature-importance ranking in
H8.

------------------------------------------------------------------------

# Research Hypotheses

The research consists of nine progressively developed hypotheses.

------------------------------------------------------------------------

# H1 — Spatial Distribution Features

## Hypothesis

> Preserving the spatial distribution of DAS energy across sensing
> channels improves physical vessel-threat discrimination compared with
> channel averaging.

H1 calculates five spatial statistics for every frequency band:

``` text
Mean
Standard Deviation
Maximum
Median
Range
```

For each frame:

``` text
250 channels × 100 frequency bands
                ↓
       5 spatial statistics
                ↓
           500 features
```

Five consecutive frames produce:

``` text
500 × 5 = 2500 features
```

## H1 Results

| Metric    |     Result |
|-----------|-----------:|
| Accuracy  | **90.54%** |
| F1        | **90.43%** |
| Threat F1 | **84.76%** |
| ROC-AUC   | **94.73%** |

------------------------------------------------------------------------

# H2 — Spectral Descriptors

## Hypothesis

> Adding compact spectral descriptors to the spatially preserved
> representation improves physical vessel-threat discrimination.

Five spectral descriptors are calculated for each frame:

``` text
Spectral Centroid
Spectral Spread
Normalized Spectral Entropy
85% Spectral Roll-off
Spectral Slope
```

H2 adds:

``` text
5 descriptors × 5 frames = 25 features
```

Resulting representation:

``` text
2500 H1 features
+ 25 H2 features
----------------
2525 features
```

## H2 Results

| Metric    |     Result |
|-----------|-----------:|
| Accuracy  | **90.58%** |
| F1        | **90.46%** |
| Threat F1 | **84.81%** |
| ROC-AUC   | **94.80%** |

------------------------------------------------------------------------

# H3 — Spatial Morphology and Coherence

## Hypothesis

> Explicitly modeling the spatial organization and adjacent-channel
> coherence of DAS energy improves physical vessel-threat detection
> beyond spatial and spectral statistics alone.

H3 models:

``` text
Spatial Centroid
Spatial Spread
Spatial Entropy
Spatial Gradient Energy
Lag-1 Spatial Autocorrelation
```

Each descriptor is summarized over:

``` text
Low-frequency region
Mid-frequency region
High-frequency region
Global Mean
Global Standard Deviation
```

This produces:

``` text
5 descriptors × 5 summaries
= 25 H3 features per frame
```

Across five frames:

``` text
25 × 5 = 125 H3 features
```

Combined H1–H3 representation:

``` text
H1 = 500 features/frame
H2 =   5 features/frame
H3 =  25 features/frame
-------------------------
     530 features/frame

530 × 5 frames = 2650 features
```

## H3 Results

| Metric         |     Result |
|----------------|-----------:|
| Accuracy       | **91.91%** |
| F1             | **91.79%** |
| Threat F1      | **86.85%** |
| ROC-AUC        | **95.36%** |
| False Alerts   |  **1,750** |
| Missed Threats |  **4,295** |

### H3 vs Baseline

``` text
Accuracy        +1.68 percentage points
Threat F1       +2.62 percentage points
ROC-AUC         +0.93 percentage points
False Alerts    −802
Missed Threats  −454
```

H3 produced the strongest improvement among the core
representation-engineering hypotheses.

------------------------------------------------------------------------

# H4 — Temporal Dynamics

## Hypothesis

> Modeling how spatial and coherence descriptors evolve across
> consecutive frames improves physical vessel-threat discrimination.

H4 models the temporal evolution of the 25 H3 descriptors across five
consecutive frames.

Six temporal statistics are calculated:

``` text
Temporal Mean
Temporal Standard Deviation
Temporal Range
Linear Temporal Slope
Mean Absolute Frame-to-Frame Change
Standard Deviation of Frame-to-Frame Change
```

This contributes:

``` text
25 descriptors × 6 statistics
= 150 temporal features
```

Total H1–H4 representation:

``` text
2650 + 150 = 2800 features
```

## H4 Results

| Metric         |     Result |
|----------------|-----------:|
| Accuracy       | **91.89%** |
| F1             | **91.78%** |
| Threat F1      | **86.87%** |
| ROC-AUC        | **95.38%** |
| False Alerts   |  **1,852** |
| Missed Threats |  **4,206** |

H4 is essentially neutral relative to H3 at the global level, but
provides the temporally enriched representation used by subsequent
experiments.

------------------------------------------------------------------------

# H5 — Distance-Aware Auxiliary Learning

## Hypothesis

> Incorporating continuous vessel-distance information as auxiliary
> supervision can improve physical-threat discrimination compared with
> treating the problem solely as binary proximity classification.

The binary task only identifies whether a vessel is inside or outside
the 1,000 m boundary.

H5 introduces a separate auxiliary distance-prediction branch.

## Leakage Prevention

Ground-truth distance is **never directly supplied to the classifier**.

Because:

``` text
Threat label = distance ≤ 1000 m
```

directly supplying the ground-truth distance would constitute target
leakage.

Instead:

``` text
DAS Representation
       ↓
Fold-contained Distance Model
       ↓
Predicted Distance
       ↓
Auxiliary Classifier Feature
```

The auxiliary branch uses the H3/H4-derived representation.

A lightweight **Ridge Regression** model generates fold-contained
predicted distances, which are then supplied as one additional
classifier feature.

H5 is treated as an **auxiliary-learning component** of the integrated
system rather than as an independently reported headline improvement.

------------------------------------------------------------------------

# H6 — Probability Calibration and Threat-Aware Decision

## Hypothesis

> Calibrating classifier probabilities and applying a threat-aware
> decision threshold can improve the reliability of physical-threat
> alerts.

XGBoost probabilities are calibrated using **Isotonic Regression**.

``` text
XGBoost Probability
        ↓
Isotonic Calibration
        ↓
Calibrated Threat Probability
        ↓
Threat-Aware Threshold
        ↓
Threat / Non-Threat Decision
```

Threshold selection is performed using validation data only.

Validation constraint:

``` text
Minimum Threat Recall = 90%
```

The threshold-selection objective prioritizes maintaining high threat
recall while reducing false proximity alerts.

## Calibration Result

``` text
Raw Brier Score        = 0.06666
Calibrated Brier Score = 0.06339
```

------------------------------------------------------------------------

# H7 — Temporal Persistence

## Hypothesis

> Requiring physical-threat predictions to persist across consecutive
> observations can suppress isolated false alarms while preserving
> sustained vessel-threat events.

Final persistence configuration:

``` text
Persistence Window = 3 observations
Required Threats   = 2
```

Therefore:

``` text
2 of 3 consecutive observations
        ↓
Persistent Threat Event
```

Temporal persistence is applied independently within each recording day.
No temporal state is carried across day boundaries.

H7 is an **operational decision layer** rather than an additional raw
feature block.

------------------------------------------------------------------------

# H8 — Feature Selection and XGBoost Refinement

## Hypothesis

> Removing redundant engineered features improves generalization and
> reduces computational complexity without sacrificing threat-detection
> performance.

The H1–H4 representation contains:

``` text
2800 engineered features
```

H8 performs fold-contained feature selection using XGBoost feature
importance.

``` text
Training Days
     ↓
XGBoost Feature Importance
     ↓
Feature Ranking
     ↓
Top-K Selection
     ↓
Fresh XGBoost Model
     ↓
Unseen Test Day
```

Evaluated feature budgets:

``` text
Top-500
Top-1000
Top-1500
```

## H8 Results

| Representation | Features |   Accuracy |  Threat F1 |    ROC-AUC | False Alerts | Missed Threats |
|----------------|---------:|-----------:|-----------:|-----------:|-------------:|---------------:|
| Top-500        |      500 |     91.81% |     86.94% |     95.89% |        1,932 |          4,178 |
| **Top-1000**   | **1000** | **91.84%** | **87.02%** | **95.92%** |    **1,918** |      **4,145** |
| Top-1500       |     1500 |     91.93% |     87.08% |     96.00% |        1,863 |          4,175 |

Top-1000 removes approximately:

``` text
64.3%
```

of the H1–H4 features while retaining comparable performance.

For the final integrated evaluation, **Top-1000 was fixed as the
selected feature budget**.

------------------------------------------------------------------------

# H9 — Higher-Order Temporal Dynamics

## Hypothesis

> Higher-order temporal derivatives of spatial/coherence features
> provide complementary information for characterizing changing physical
> DAS disturbances.

H9 models first- and second-order temporal changes using:

``` text
Mean Absolute First Difference
Maximum Absolute First Difference
Mean Absolute Second Difference
Maximum Absolute Second Difference
```

H9 contributes:

``` text
100 additional features
```

producing:

``` text
2800 + 100 = 2900 features
```

## H9 Results

| Metric         |     Result |
|----------------|-----------:|
| Accuracy       | **91.91%** |
| Threat F1      | **87.05%** |
| Threat Recall  | **83.52%** |
| ROC-AUC        | **96.00%** |
| False Alerts   |  **1,845** |
| Missed Threats |  **4,210** |

------------------------------------------------------------------------

# Final Integrated Model

The final system integrates the research contributions in a staged
architecture rather than blindly concatenating every experiment.

``` text
Original DAS Data
        ↓
Original Preprocessing
        ↓
H1 Spatial Features
        ↓
H2 Spectral Features
        ↓
H3 Spatial Morphology + Coherence
        ↓
H4 Temporal Dynamics
        ↓
H9 Higher-Order Temporal Dynamics
        ↓
2900-Dimensional Representation
        ↓
H8 Fold-Contained Feature Selection
        ↓
Top-1000 Features
        ↓
H5 Fold-Contained Distance Prediction
        ↓
XGBoost Classification
        ↓
H6 Isotonic Probability Calibration
        ↓
Threat-Aware Decision Threshold
        ↓
H7 2-of-3 Temporal Persistence
        ↓
FINAL PHYSICAL THREAT ALERT
```

### Final Architecture

> **H1 + H2 + H3 + H4 + H9 → H8 → H5 → XGBoost → H6 → H7**

------------------------------------------------------------------------

# Final Model Configuration

| Component               | Configuration                 |
|-------------------------|-------------------------------|
| Base representation     | H1 + H2 + H3 + H4 + H9        |
| Base dimensionality     | **2,900**                     |
| Feature selection       | H8                            |
| Selected features       | **Top-1,000**                 |
| Main classifier         | **XGBoost**                   |
| Auxiliary model         | **Ridge Regression**          |
| Probability calibration | **Isotonic Regression**       |
| Threat class            | **Class 0**                   |
| Threat boundary         | **1,000 m**                   |
| Persistence window      | **3 observations**            |
| Persistence requirement | **2 of 3**                    |
| Cross-validation        | **10-fold leave-one-day-out** |
| Random seed             | **42**                        |
| XGBoost estimators      | **500**                       |
| XGBoost learning rate   | **0.05**                      |
| XGBoost max depth       | **10**                        |
| XGBoost tree method     | **hist**                      |
| XGBoost parallel jobs   | **8**                         |

------------------------------------------------------------------------

# Final Integrated Results

The final H1–H9 system was evaluated across the complete **10-fold
leave-one-day-out** protocol.

## Headline Performance

| Metric                     | Final Result |
|----------------------------|-------------:|
| **Accuracy**               |   **91.28%** |
| **Threat Precision**       |   **87.66%** |
| **Threat Recall**          |   **85.12%** |
| **Threat F1**              |   **86.37%** |
| **Non-Threat F1**          |   **93.59%** |
| **Macro F1**               |   **89.98%** |
| **Weighted F1**            |   **91.25%** |
| **ROC-AUC**                |   **95.69%** |
| **PR-AUC**                 |   **94.12%** |
| **Raw Brier Score**        |  **0.06666** |
| **Calibrated Brier Score** |  **0.06339** |

------------------------------------------------------------------------

# Final Threat Detection

Final confusion matrix:

``` text
                         Predicted
                    Threat   Non-Threat
Actual Threat        20,655      3,610
Actual Non-Threat     2,908     47,594
```

Therefore:

``` text
True Positives       = 20,655
Missed Threats       =  3,610
False Alerts         =  2,908
True Negatives       = 47,594
```

Final threat-oriented performance:

``` text
Threat Precision = 87.66%
Threat Recall    = 85.12%
Threat F1        = 86.37%
```

------------------------------------------------------------------------

# Baseline vs Final Model

The original channel-averaged XGBoost classifier is used as the
baseline.

| Metric         | Baseline | Final H1–H9 |       Change |
|----------------|---------:|------------:|-------------:|
| Accuracy       |   90.23% |  **91.28%** | **+1.05 pp** |
| Threat Recall  |   80.42% |  **85.12%** | **+4.70 pp** |
| Threat F1      |   84.23% |  **86.37%** | **+2.14 pp** |
| ROC-AUC        |   94.43% |  **95.69%** | **+1.26 pp** |
| Missed Threats |    4,749 |   **3,610** |   **−1,139** |
| False Alerts   |    2,552 |       2,908 |         +356 |

The final model reduces missed threats:

``` text
4,749 → 3,610
```

This corresponds to:

``` text
1,139 fewer missed threats
≈ 24.0% reduction
```

This improvement is accompanied by an increase in false proximity
alerts:

``` text
2,552 → 2,908
```

The final model should therefore be interpreted as a stronger
threat-oriented operating point rather than a universally optimal
classifier at every possible decision threshold.

------------------------------------------------------------------------

# H1–H9 Experimental Summary

| Hypothesis | Main Contribution              | Primary Role                     |
|------------|--------------------------------|----------------------------------|
| **H1**     | Spatial distribution           | Spatial representation           |
| **H2**     | Spectral descriptors           | Spectral representation          |
| **H3**     | Spatial morphology & coherence | Core representation contribution |
| **H4**     | Temporal dynamics              | Temporal representation          |
| **H5**     | Distance-aware learning        | Auxiliary supervision            |
| **H6**     | Probability calibration        | Decision reliability             |
| **H7**     | Temporal persistence           | Event-level decision logic       |
| **H8**     | Feature selection              | Model refinement / efficiency    |
| **H9**     | Higher-order temporal dynamics | Temporal refinement              |

------------------------------------------------------------------------

# Key Research Findings

### 1. Spatial structure is informative

H1 demonstrates that channel averaging discards useful spatial
information contained in DAS measurements.

### 2. Spectral information is complementary

H2 provides an incremental improvement, indicating that spectral
characteristics contain information beyond spatial statistics.

### 3. Spatial morphology and coherence provide the strongest core improvement

H3 produces the largest improvement among the core representation
hypotheses, particularly reducing false proximity alerts.

### 4. Temporal information provides additional context

H4 captures how spatial and coherence features evolve across consecutive
observations.

### 5. Feature redundancy is substantial

H8 demonstrates that a large fraction of engineered features can be
removed while retaining comparable detection performance.

### 6. Higher-order temporal information is useful

H9 introduces first- and second-order temporal changes and improves
discrimination relative to H4.

### 7. Probability calibration improves reliability

The Brier score decreases from:

``` text
Raw Brier Score        = 0.06666
Calibrated Brier Score = 0.06339
```

### 8. Temporal persistence converts predictions into events

H7 introduces an operational layer in which sustained physical
disturbances are distinguished from isolated model predictions.

------------------------------------------------------------------------

# Leakage Prevention

A central methodological requirement of this research is preventing
target leakage.

The ground-truth vessel distance is **not** directly provided to the
binary classifier.

Instead, H5 generates:

``` text
Training DAS Data
       ↓
Fold-contained distance model
       ↓
Predicted distance
       ↓
Binary classifier
```

For the integrated evaluation:

- Outer test days remain completely unseen during model fitting.
- Auxiliary distance predictions are generated fold-contained.
- Feature selection is performed within the corresponding training data.
- Calibration is fitted using validation predictions.
- Decision thresholds are selected using validation data.
- Temporal persistence is applied only to chronological test-day
  predictions.
- No temporal persistence state crosses day boundaries.

------------------------------------------------------------------------

# Computational Design

The repository uses a memory-conscious preprocessing pipeline for the
large DAS feature matrix.

The raw feature matrix contains:

``` text
74,771 × 250 × 100
```

features.

The experimental implementation uses `float32` representation for the
loaded DAS matrix to reduce memory consumption while preserving the
numerical structure required for the experiments.

Feature extraction is performed progressively and the original
high-dimensional matrix is released after reduction where possible.

------------------------------------------------------------------------

# Reproducibility

The repository contains the implementation and experimental artifacts
required to reproduce the research pipeline.

The original DAS dataset is **not included in this repository**.

After obtaining the dataset from its official distribution source, place
the required HDF5 file under:

``` text
data/
```

Expected filename:

``` text
dataset_sensor_range_1440_1690_0.h5
```

Frequency-band definition:

``` text
data/fbands.csv
```

------------------------------------------------------------------------

# Installation

Clone the repository:

``` bash
git clone https://github.com/GaurishBudhiraja/DAS-Physical-Threat-Detection.git
cd DAS-Physical-Threat-Detection
```

Create a Python virtual environment:

``` bash
python3 -m venv venv311
source venv311/bin/activate
```

Install dependencies:

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Running the Experiments

## Baseline

``` bash
bash scripts/run_xgb_classif_baseline_all_folds-best.sh
```

## H5 — Distance-Aware Learning

``` bash
python scripts/run_h5_distance_aware.py
```

## H6 — Probability Calibration

``` bash
python scripts/run_h6_probability_calibration.py
```

## H7 — Temporal Persistence

``` bash
python scripts/run_h7_temporal_persistence.py
```

## H5 + H6 + H7 Integrated Stage

``` bash
python scripts/run_h5_h6_h7_final.py
```

## H8 — Feature Selection

``` bash
python scripts/run_h8_feature_selection.py
```

## H9 — Higher-Order Temporal Dynamics

``` bash
python scripts/run_h9_multiscale_temporal.py
```

## Final H1–H9 Integrated Model

``` bash
python scripts/run_final_h1_h9_model.py
```

The final integrated experiment performs the complete outer **10-fold
leave-one-day-out** evaluation.

------------------------------------------------------------------------

# Repository Structure

``` text
DAS-Physical-Threat-Detection/
│
├── data/
│   └── fbands.csv
│
├── models/
│   ├── baseline_xgb_classification_model.py
│   └── baseline_xgb_regression_model.py
│
├── src/
│   ├── data_splitter.py
│   ├── data_splitter_hdf5.py
│   ├── hdf5_data_loader.py
│   ├── load_and_split_dataset.py
│   ├── model_experiment_hdf5.py
│   └── ...
│
├── scripts/
│   ├── run_xgb_classif_baseline_all_folds-best.sh
│   ├── run_h5_distance_aware.py
│   ├── run_h6_probability_calibration.py
│   ├── run_h7_temporal_persistence.py
│   ├── run_h5_h6_h7_final.py
│   ├── run_h8_feature_selection.py
│   ├── run_h9_multiscale_temporal.py
│   └── run_final_h1_h9_model.py
│
├── research/
│   ├── baseline/
│   ├── h1_metrics.csv
│   ├── h2_metrics.csv
│   ├── h3_metrics.csv
│   ├── h4_metrics.csv
│   ├── h8_feature_importance_fold_*.csv
│   ├── final_feature_importance_fold_*.csv
│   └── ...
│
├── results/
│   └── ...
│
├── docs/
│   └── ...
│
├── DATA_NOTES.md
├── EXPERIMENT_LOG.md
├── LITERATURE_MATRIX.md
├── METHODOLOGY.md
├── RESEARCH_PLAN.md
├── requirements.txt
├── LICENSE
└── README.md
```

------------------------------------------------------------------------

# Research Artifacts

The repository contains experiment outputs for:

``` text
Baseline
H1
H2
H3
H4
H5 + H6 + H7
H8
H9
Final Integrated H1–H9 Model
```

Final research artifacts include:

- Fold-level metrics
- Per-day evaluation results
- Final predictions
- Feature-importance rankings
- Confusion-matrix results
- Calibration metrics
- Model summaries
- Research visualizations

These artifacts are retained to support analysis and reproducibility of
the reported experiments.

------------------------------------------------------------------------

# Research Visualizations

The repository contains generated visualizations associated with the
research experiments, including analysis of:

- Model performance
- Fold-level behavior
- Feature importance
- Threat detection performance
- Final model behavior

------------------------------------------------------------------------

# Limitations

This project represents a **research prototype**, not a
production-certified submarine cable monitoring system.

Important limitations include:

- Evaluation is based on a specific DAS dataset and cable segment.
- The observation period is limited.
- Generalization to other submarine cables requires additional
  validation.
- Vessel-distance labels depend on the underlying dataset and associated
  vessel information.
- The physical-threat boundary is fixed at 1,000 m for this study.
- Environmental conditions can influence DAS measurements.
- False proximity alerts remain present.
- The final model has not been validated in an operational deployment.
- Cross-cable and cross-domain generalization remain open research
  problems.
- The current study evaluates physical vessel threats rather than
  arbitrary underwater or cable-interaction events.

The reported results should therefore be interpreted as **experimental
evidence on the evaluated dataset**, rather than a guarantee of
identical performance in operational environments.

------------------------------------------------------------------------

# Research Contribution

The principal contribution of this research is an integrated
physical-threat detection framework that extends conventional
channel-averaged DAS classification by modeling:

``` text
Spatial Distribution
        +
Spectral Characteristics
        +
Spatial Morphology
        +
Adjacent-Channel Coherence
        +
Temporal Dynamics
        +
Higher-Order Temporal Changes
        +
Distance-Aware Auxiliary Information
        +
Feature Selection
        +
Probability Calibration
        +
Temporal Persistence
```

The strongest representation-level improvement is obtained through **H3
spatial morphology and coherence features**, while the complete H1–H9
pipeline combines representation engineering, feature refinement,
auxiliary supervision, calibrated probability estimation, and temporal
decision logic.

------------------------------------------------------------------------

# Research Foundation and Attribution

This research uses an existing open-source DAS vessel-detection
implementation and publicly released DAS dataset as the technical and
reproducibility foundation.

The present repository substantially extends that foundation through:

- The H1–H9 research framework
- Spatial feature engineering
- Spectral feature engineering
- Spatial morphology descriptors
- Channel-coherence descriptors
- Temporal feature engineering
- Higher-order temporal dynamics
- Distance-aware auxiliary learning
- Fold-contained feature selection
- Probability calibration
- Threat-aware threshold selection
- Temporal persistence
- Integrated 10-fold evaluation
- Research-specific experiment scripts
- Research-specific analysis artifacts

Original open-source implementation used as the starting technical
foundation:

https://github.com/UAH-PSI/das-vessel-detection

Underlying dataset:

**Marlinks-NS DAS Dataset**

Dataset DOI:

**10.5281/zenodo.15611778**

Associated publication DOI:

**10.1109/JSTARS.2026.3716768**

Associated preprint DOI:

**10.48550/arXiv.2607.28306**

The original implementation, dataset, and associated publication should
be appropriately cited when their work is used or referenced.

------------------------------------------------------------------------

# Academic Citation

If referencing this research implementation:

``` text
Gaurish Budhiraja,
"DAS Physical Threat Detection for Submarine Optical-Fiber Cables:
A Spatiotemporal Machine-Learning Framework for Physical Vessel-Threat
Detection Using Distributed Acoustic Sensing."
Research Implementation, 2026.
```

For the underlying DAS dataset and original vessel-detection
methodology, please additionally cite the original dataset and
publication identified above.

------------------------------------------------------------------------

# License

This repository is distributed under the:

**GNU General Public License v3.0 (GPLv3)**

See:

``` text
LICENSE
```

Dataset licensing remains subject to the terms of the original dataset
distribution.

------------------------------------------------------------------------

# Author

## Gaurish Budhiraja

**B.Tech Computer Science and Engineering**  
**Vellore Institute of Technology**  
**Chennai, India**

### Research Interests

``` text
Distributed Acoustic Sensing
Machine Learning
Submarine Cable Protection
Physical Threat Detection
Signal Processing
Cybersecurity
AI Systems
```

### GitHub

https://github.com/GaurishBudhiraja

------------------------------------------------------------------------

<p align="center">

## DAS Physical Threat Detection

### H1–H9 Spatiotemporal Research Framework

**Machine-learning-based physical vessel-threat detection using
Distributed Acoustic Sensing.**

</p>
