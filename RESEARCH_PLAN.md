# Phase 1 Research Plan
## Self-Supervised DAS-Based Physical Threat Detection for Submarine and Buried Fiber-Optic Cables

---

## 1. Research Objective

This project investigates whether self-supervised spatiotemporal representation learning on real Distributed Acoustic Sensing (DAS) recordings can improve detection of physical threats and anomalous physical events affecting fiber-optic cables when security-labeled data are scarce.

The primary focus of Phase 1 is physical-layer threat detection, including:

- Vessel activity near submarine cables
- Potential anchor/drag-related activity
- Fiber manipulation
- Physical tampering
- Other abnormal physical events detectable through DAS

The research will evaluate both known-event classification and detection of previously unseen physical events.

The central research question is:

> Can representations learned from large quantities of real, unlabeled or differently-labeled DAS recordings improve physical-threat detection when only a small amount of security-relevant labeled data is available?

---

## 2. Research Hypothesis

The primary hypothesis is:

> Self-supervised pretraining on real DAS recordings will produce representations that improve downstream physical-threat detection compared with models trained entirely from scratch, particularly when labeled security data are limited.

Secondary hypotheses:

1. Self-supervised pretraining will improve performance under limited labeled-data conditions.
2. Spatiotemporal representations will generalize better across different DAS deployments than purely supervised models.
3. A pretrained representation can support anomaly detection for physical event types that were not present during supervised training.
4. The benefit of self-supervised learning will depend on the similarity between the pretraining domain and downstream security domain.
5. More complex architectures should only be retained when experimental evidence demonstrates a measurable improvement over simpler baselines.

---

## 3. Scope of Phase 1

### Included

- Distributed Acoustic Sensing (DAS)
- Physical-layer events
- Fiber manipulation
- Vessel/vehicle activity
- Physical intrusion analogues
- Anomaly detection
- Supervised classification
- Self-supervised learning
- Masked reconstruction
- Spatiotemporal representation learning
- Cross-domain evaluation
- Label-efficiency evaluation
- Robustness evaluation

### Not Included in Phase 1

The following are outside the implementation scope of Phase 1:

- Optical-layer SoP attacks
- Optical signal interception
- Network-layer attacks
- Landing-station IDS
- NMS/SCADA attacks
- Application-layer attacks
- Cyber intrusion detection using IP/network traffic

These may become subsequent phases of the broader submarine-cable security research program.

---

# 4. Dataset Strategy

## 4.1 Primary Security-Labeled Dataset

### Tomasov et al. — Scientific Data 2025

Paper:

> Tomasov et al., "Comprehensive Dataset for Event Classification Using Distributed Acoustic Sensing (DAS) Systems"

DOI:

10.1038/s41597-025-05088-4

Dataset DOI:

10.6084/m9.figshare.27004732

Role:

Primary labeled DAS dataset for physical-event classification.

Reported event categories include:

- Walking
- Running
- Driving / vehicular movement
- Longboarding
- Fence climbing
- Fiber manipulation

The security-relevant classes are particularly important because they provide real physical intrusion-related signals.

The dataset is terrestrial/buried fiber rather than submarine fiber.

---

## 4.2 Submarine Security-Relevant Dataset

### Marlinks-NS

Relevant work:

Ramirez-Torres et al.

arXiv:

2509.11614

2607.28306

Role:

Submarine-cable domain evaluation.

The currently accessible release is a reduced demonstration dataset rather than the complete 10-day recording.

The full dataset release should be treated as pending until the official open release is confirmed.

The currently available material may be used for pipeline development and preliminary evaluation.

---

## 4.3 Self-Supervised Pretraining Data

Large real DAS datasets may be used as unlabeled or differently-labeled pretraining data.

Potential sources:

### PubDAS

Role:

Large-scale DAS representation pretraining and cross-domain evaluation.

Important characteristic:

Contains real DAS recordings, including seafloor deployments, but is primarily geophysical/seismological rather than security-labeled.

---

### SUBMERSE

Role:

Submarine DAS pretraining.

The project provides large real submarine DAS earthquake datasets.

Earthquake labels are not required for the self-supervised phase because the pretraining objective does not depend on security labels.

---

### OOI Regional Cabled Array DAS

Role:

Submarine DAS representation pretraining and domain-specific evaluation.

The 2025–2026 multi-span DAS release contains real submarine cable recordings.

Only manageable subsets should initially be downloaded.

The entire multi-terabyte corpus should NOT be downloaded unnecessarily.

---

## 4.4 Supplementary Submarine Dataset

### Trondheim DAS

Zenodo record:

18851481

Role:

Potential supplementary evaluation/generalization dataset.

It contains real ocean-bottom DAS data associated with submarine-cable ship tracking research.

Its actual available size and structure must be inspected before incorporating it into the experimental pipeline.

---

# 5. Dataset Principle

No dataset will be treated as useful merely because it is large.

Every dataset must be characterized by:

- Sensor/interrogator
- Fiber type
- Deployment environment
- Sampling frequency
- Spatial sampling/channel spacing
- Temporal resolution
- Recording duration
- Signal representation
- Label structure
- Event definitions
- Geographic environment
- Data availability
- License
- Preprocessing already applied
- Potential train/test leakage
- Relevance to the research question

Dataset assumptions must never be substituted for actual inspection.

---

# 6. Phase 1 Architecture

The architecture will NOT be selected solely because a particular model is currently popular.

Models will be compared experimentally.

The candidate hierarchy is:

## Level 1 — Classical Baselines

- Statistical threshold detector
- Isolation Forest
- One-Class SVM
- Random Forest
- XGBoost

Feature candidates:

- Signal energy
- RMS amplitude
- Peak amplitude
- Dominant frequency
- Spectral entropy
- Band energy
- Frequency-band ratios
- Spectral centroid
- Other features justified by dataset analysis

---

## Level 2 — Supervised Deep Learning

Candidate models:

- CNN
- CNN-LSTM

Purpose:

Establish a supervised deep-learning baseline before introducing self-supervised learning.

---

## Level 3 — Reconstruction-Based Detection

Candidate:

- CNN Autoencoder

Purpose:

Learn normal signal structure and identify anomalous events using reconstruction error.

---

## Level 4 — Self-Supervised Learning

Primary candidate:

- Masked Autoencoder / masked reconstruction model

Candidate encoder families:

1. CNN
2. Transformer
3. CNN + Transformer hybrid

The final architecture will be selected through controlled experiments.

---

# 7. Self-Supervised Learning Strategy

The main self-supervised objective will be masked reconstruction.

A DAS recording will be represented as a spatiotemporal signal:

```text
Time
  ↑
  │
  │
  │   DAS waterfall
  │
  └────────────────→ Distance / Channel
