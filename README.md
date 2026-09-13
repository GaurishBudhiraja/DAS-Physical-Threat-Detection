# DAS Physical Threat Detection for Submarine Optical-Fiber Cables

### A Spatiotemporal Machine-Learning Framework for Physical Vessel-Threat Detection Using Distributed Acoustic Sensing

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![DAS](https://img.shields.io/badge/Sensing-DAS-6f42c1)
![Research](https://img.shields.io/badge/Research-IEEE%20Style-red)
![Status](https://img.shields.io/badge/Status-Research%20Complete-success)
![License](https://img.shields.io/badge/License-GPLv3-blue)

</p>

---

## Author

**Gaurish Budhiraja**  
B.Tech Computer Science and Engineering  
Vellore Institute of Technology, Chennai, India

**Research Area:** Distributed Acoustic Sensing · Submarine Cable Protection · Machine Learning · Physical Threat Detection · Spatiotemporal Signal Analysis

GitHub: [GaurishBudhiraja](https://github.com/GaurishBudhiraja)

---

# Overview

Submarine optical-fiber cables form a critical component of global communications infrastructure. Although these cables are designed primarily for data transmission, the optical fiber itself can also act as a distributed sensing medium.

**Distributed Acoustic Sensing (DAS)** transforms an optical fiber into a dense array of virtual acoustic sensing locations. Mechanical disturbances near the cable generate detectable changes in the optical signal, making DAS suitable for continuous monitoring of physical activity around submarine cables.

This research develops and evaluates a machine-learning framework for **physical vessel-threat detection around submarine optical-fiber cables**.

The central objective is to determine whether the spatial, spectral, morphological, and temporal structure contained in DAS measurements can be exploited to improve detection of vessels operating within a defined proximity to the cable.

The research is organized into **nine hypotheses (H1–H9)** and culminates in an integrated machine-learning pipeline combining:

- Spatial feature engineering
- Spectral characterization
- Spatial morphology and channel coherence
- Temporal dynamics
- Higher-order temporal derivatives
- Fold-contained feature selection
- Distance-aware auxiliary learning
- Probability calibration
- Threat-aware decision making
- Temporal persistence

The final system is evaluated using **10-fold leave-one-day-out cross-validation**, where each complete recording day is held out as an unseen test set.

---

# Research Objective

The primary research question is:

> **Can engineered spatiotemporal representations of Distributed Acoustic Sensing measurements improve the detection of physical vessel threats around submarine optical-fiber cables compared with a conventional channel-averaged XGBoost baseline?**

The research focuses exclusively on **physical sensing and vessel-threat detection**.

It does **not** attempt to detect network-layer attacks such as:

- Man-in-the-middle attacks
- Packet injection
- Network sniffing
- Traffic interception
- Protocol attacks

The target phenomenon is a **physical vessel presence/proximity event detected through DAS measurements**.

---

# System Concept

The complete research pipeline follows:

```text
                    SUBMARINE DAS DATA
                           │
                           ▼
              Original DAS Preprocessing
                           │
                           ▼
                 5-Frame Temporal Window
                           │
                           ▼
              ┌─────────────────────────┐
              │ H1 Spatial Features     │
              │ H2 Spectral Features    │
              │ H3 Spatial Coherence    │
              └────────────┬────────────┘
                           │
                           ▼
                  H4 Temporal Dynamics
                           │
                           ▼
               H9 Higher-Order Dynamics
                           │
                           ▼
                H8 Feature Selection
                    Top-1000 Features
                           │
                           ├───────────────┐
                           │               │
                           ▼               ▼
                    XGBoost Input    H5 Auxiliary
                                     Distance Branch
                           │               │
                           └───────┬───────┘
                                   ▼
                             XGBoost Classifier
                                   │
                                   ▼
                        H6 Probability Calibration
                                   │
                                   ▼
                       Threat-Aware Thresholding
                                   │
                                   ▼
                       H7 Temporal Persistence
                                   │
                                   ▼
                         PHYSICAL THREAT ALERT
