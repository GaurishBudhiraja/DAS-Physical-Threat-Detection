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

Submarine optical-fiber cables form a critical component of global communications infrastructure. Although these cables are primarily designed for data transmission, the optical fiber can also function as a distributed sensing medium.

**Distributed Acoustic Sensing (DAS)** transforms an optical fiber into a dense array of virtual acoustic sensing locations. Mechanical disturbances occurring near the cable generate measurable changes in the optical signal, allowing the fiber to continuously sense physical activity along its route.

This repository presents a machine-learning framework for **physical vessel-threat detection around submarine optical-fiber cables using Distributed Acoustic Sensing**.

The research investigates whether the spatial, spectral, morphological, coherence-based, and temporal structure contained within DAS measurements can improve detection of vessels operating within a defined proximity to a submarine cable.

The research is organized into **nine experimental hypotheses (H1–H9)** and culminates in an integrated physical-threat detection pipeline combining:

- Spatial feature engineering
- Spectral characterization
- Spatial morphology
- Adjacent-channel coherence
- Temporal dynamics
- Higher-order temporal dynamics
- Feature selection
- Distance-aware auxiliary learning
- Probability calibration
- Threat-aware decision making
- Temporal persistence

The final system is evaluated using **10-fold leave-one-day-out cross-validation**, with each complete recording day held out as an unseen test fold.

---

# Research Objective

The primary research question is:

> **Can engineered spatiotemporal representations of Distributed Acoustic Sensing measurements improve the detection of physical vessel threats around submarine optical-fiber cables compared with a conventional channel-averaged XGBoost baseline?**

The research focuses exclusively on the **physical sensing layer**.

This project does not attempt to detect network-layer attacks such as:

- Man-in-the-middle attacks
- Packet injection
- Network sniffing
- Protocol attacks
- Network traffic manipulation

The target phenomenon is a **physical vessel presence/proximity event detected through DAS measurements**.

---

# System Architecture

The final research system follows a staged architecture:

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
             XGBoost Features          H5 Distance Branch
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
