# Literature Matrix — Phase 1

## Research Focus

Self-supervised spatiotemporal representation learning for
physical-threat and anomaly detection using Distributed Acoustic Sensing
(DAS), with particular emphasis on label scarcity, security-relevant
events, submarine-cable applicability, unseen-event detection, and
cross-domain transfer.

---

## Core Literature

| Paper | Year / Venue | DAS Modality & Environment | Primary Task | Dataset / Data | Method / Model | Main Contribution | Limitation Relevant to Our Work | Role in Our Research |
|---|---|---|---|---|---|---|---|---|
| Tomasov et al. — *Comprehensive Dataset for Event Classification Using Distributed Acoustic Sensing (DAS) Systems* | 2025, Scientific Data | DAS; buried terrestrial optical fiber around a university campus | Multi-class event classification | Labeled DAS recordings containing ordinary activities and security-relevant events | CNN-based event classification | Provides an openly documented labeled DAS dataset containing physical events and potential security threats | Supervised event classification; terrestrial deployment; does not investigate self-supervised pretraining, label-efficiency scaling, or unseen-event transfer | Primary security-event dataset and supervised baseline reference |
| Duan, Chen & He — *DAS-MAE: A self-supervised pre-training framework for universal and high-performance representation learning of distributed fiber-optic acoustic sensing* | 2025, IEEE/OSA JLT / arXiv | DAS spatiotemporal signals | Representation learning and few-shot downstream classification | DAS recordings used for self-supervised pretraining and downstream evaluation | DAS-MAE; masked reconstruction / masked autoencoder | Demonstrates self-supervised representation learning for DAS and reports strong few-shot performance | Demonstrated for DAS event recognition rather than our security-focused transfer setting; does not establish transfer between terrestrial security events and submarine cable threats | Primary methodological reference for masked-reconstruction SSL |
| Ding et al. — *DASFormer: self-supervised pretraining for earthquake monitoring* | 2025, Visual Intelligence | DAS; seismic/telecommunication-fiber environments | Unsupervised earthquake / seismic anomaly detection and downstream tasks | Real DAS data including Ridgecrest and OOI-related data | DASFormer; coarse-to-fine spatiotemporal self-supervised architecture based on U-Net components | Shows that unlabeled DAS can be pretrained to learn spatial-temporal representations and used for anomaly detection | Downstream validation is centered on earthquake/seismic monitoring rather than security-threat classification | Architectural and methodological reference for spatiotemporal SSL |
| Ramirez-Torres et al. — *A Distributed Acoustic Sensing Dataset for Vessel Detection and Localization in Submarine Cable Protection* (Marlinks-NS) | 2026, arXiv | DAS; buried submarine telecommunications fiber in the North Sea | Vessel detection and vessel-to-cable distance estimation | 74,771 labeled instances from 10 days of recording along a 2,554 m segment of a 28 km buried cable; processed spectral-energy features from 250 sensing channels | Machine-learning classification/regression using processed DAS features | Provides submarine-cable DAS data specifically oriented toward cable protection and vessel monitoring | Focuses on supervised vessel-related tasks; does not investigate self-supervised pretraining, label efficiency, unseen threat classes, or cross-domain SSL transfer | Primary submarine security-relevant reference and downstream evaluation candidate |
| Spica et al. — *PubDAS* | 2023, Seismological Research Letters | Multi-site DAS including seafloor/seismic deployments | DAS data repository / seismic research | Large multi-site DAS repository | Dataset/resource rather than a single security model | Provides broad DAS data useful for research and representation learning | Data are primarily geophysical/environmental rather than security-labeled | Candidate source for unlabeled/self-supervised pretraining |
| SUBMERSE project datasets | 2023–2026 project | Submarine DAS / seafloor fiber | Earthquake and seismic monitoring | Large submarine DAS earthquake collection | Project-specific seismic models including DeepDAS | Provides substantial real submarine DAS data | Labels are seismic/environmental rather than security-threat labels | Candidate submarine-domain pretraining source |
| OOI Regional Cabled Array DAS data | 2025–2026 releases | Real submarine fiber / ocean-bottom cable | Seismic and ocean/environmental monitoring | Multi-span DAS recordings from OOI infrastructure | Dataset/resource | Provides real submarine DAS recordings suitable for representation learning | Not security-labeled; access, subset size, and exact usable recordings must be documented before experimentation | Candidate real submarine pretraining/generalization source |
| Trondheim DAS dataset / related ship-tracking work | Recent | Real submarine/seafloor DAS | Ship / ocean activity monitoring | Trondheim-fjord submarine DAS data | DAS-based ship/event analysis | Provides another real submarine DAS environment with physical marine activity | Dataset contents, labels, license, and exact accessible recordings must be inspected before use | Optional supplementary submarine-domain dataset |
| Malik et al. — MDPI *AI* | 2025 | State-of-polarization (SoP), not DAS | Multi-threat / fiber physical-security detection | Fiber-optic sensing data | ML-based threat classification | Demonstrates physical-threat detection using another fiber sensing modality and provides a useful performance reference | Different sensing modality; results are not directly comparable to DAS | Contextual reference and external performance sanity check |

---

## Detailed Research Gaps

### Gap 1 — Self-supervised DAS learning has not been established for security-oriented physical-threat detection

DAS-MAE demonstrates masked-reconstruction self-supervised representation
learning for DAS, while DASFormer demonstrates self-supervised
spatiotemporal learning primarily in seismic/earthquake monitoring.

Our research investigates whether these learned representations transfer
to physical-security events such as vessel activity, fiber manipulation,
and other intrusion-relevant events under limited labeled data.

---

### Gap 2 — Label scarcity has not been systematically evaluated for DAS security events

Existing security-oriented DAS work relies substantially on labeled
examples for downstream event classification.

Our study will explicitly evaluate performance as the amount of security
labeled data is reduced.

Planned label fractions:

- 100%
- 25%
- 10%
- 1%

The objective is to determine whether self-supervised pretraining provides
a measurable advantage when security labels are scarce.

---

### Gap 3 — Unseen physical events / open-set detection

Most event-classification systems evaluate classes that were represented
during training.

Our research will investigate whether a representation learned from
normal and diverse DAS recordings can identify an event type for which
the classifier has not received labeled training examples.

Planned experiment:

**Leave-One-Event-Type-Out (LOETO)**

One physical event type is completely excluded from supervised training
and treated as an unknown event during evaluation.

---

### Gap 4 — Terrestrial-to-submarine domain transfer

Tomasov provides labeled terrestrial DAS security-relevant events, while
Marlinks-NS provides real submarine-cable DAS data associated with vessel
activity.

This creates an opportunity to measure domain shift rather than assuming
that a model trained on one deployment transfers automatically to another.

Planned directions:

1. Pretrain → terrestrial fine-tuning
2. Pretrain → submarine fine-tuning
3. Terrestrial → submarine transfer
4. Submarine → terrestrial transfer

The experiment will report both absolute performance and performance
degradation under domain transfer.

---

### Gap 5 — Representation choice is not assumed a priori

Rather than declaring that spectrograms, PSDs, or raw DAS signals are
optimal before experimentation, the study will compare representations
under controlled conditions.

Candidate representations:

1. Raw time-domain DAS windows
2. Spectrogram representation
3. Power spectral density / spectral features

The same evaluation protocol will be used wherever possible.

---

### Gap 6 — Architecture choice will be experimentally justified

The research will compare:

- Classical feature-based models
- CNN
- CNN-LSTM
- CNN autoencoder
- Transformer
- CNN + Transformer
- DAS-MAE-style masked reconstruction
- DASFormer-inspired spatiotemporal SSL

The final architecture will be selected based on experimental evidence
rather than assuming that a Transformer is automatically superior.

---

## Planned Experimental Questions

### E1 — Classical / supervised baseline

How well can conventional machine-learning and supervised deep-learning
methods classify physical DAS events?

### E2 — Anomaly detection

Can normal-event representations be used to detect anomalous physical
events without requiring examples of every threat?

### E3 — Representation comparison

Which DAS representation provides the strongest downstream performance?

### E4 — Architecture comparison

Which architecture provides the best accuracy/efficiency trade-off?

### E5 — Self-supervised learning benefit

Does self-supervised pretraining outperform training the same architecture
from scratch?

### E6 — Label efficiency

How rapidly does performance degrade as security labels become scarce?

### E7 — Unseen-event detection

Can the system detect an event type absent from supervised training?

### E8 — Cross-domain transfer

Can representations learned from one DAS deployment transfer to another,
particularly between terrestrial and submarine environments?

### E9 — Robustness

How robust is the system to:

- sensor/channel dropout
- noise
- missing measurements
- different window sizes
- sampling-rate changes
- environmental variation

---

## Planned Baselines

### Classical

- Statistical threshold detector
- Isolation Forest
- One-Class SVM
- Random Forest
- XGBoost

### Deep supervised

- CNN
- CNN-LSTM

### Reconstruction / anomaly detection

- CNN Autoencoder
- Reconstruction-error detector

### Self-supervised

- DAS-MAE-style masked reconstruction
- DASFormer-inspired spatiotemporal SSL

---

## Planned Evaluation Metrics

### Classification

- Macro-F1
- Weighted-F1
- Precision
- Recall
- ROC-AUC
- PR-AUC

### Anomaly Detection

- AUROC
- AUPRC
- False-positive rate
- False-negative rate
- Detection latency

### Label Efficiency

Performance versus:

- 100% labels
- 25% labels
- 10% labels
- 1% labels

### Domain Transfer

Report:

- In-domain F1
- Cross-domain F1
- Absolute performance degradation
- Relative performance degradation

---

## Evidence Status

The following distinction will be maintained throughout the project:

**Verified**

Claims directly supported by the original paper, official dataset
repository, DOI record, or official data-access page.

**To Be Verified**

Claims for which the existence of the resource is established but the
exact dataset contents, access conditions, license, size, or usable subset
have not yet been independently inspected.

**Not Verified**

Dataset names, sizes, performance figures, or other claims for which an
authoritative source has not been located.

No unverified numerical claim will be used in the final paper.

---

## Current Novelty Hypothesis

The central research hypothesis is:

> Self-supervised spatiotemporal representation learning from abundant
> DAS recordings can improve physical-threat detection under severe label
> scarcity and improve generalization to deployment domains and event
> types not represented in supervised training.

The hypothesis will be considered supported only if the experimental
results demonstrate statistically and practically meaningful improvement
over appropriately matched supervised and classical baselines.

---

## Important Scope Boundary

This phase concerns **physical-layer environmental and infrastructure
events detected through DAS**.

Examples include:

- vessel activity near submarine cable
- fiber manipulation
- physical intrusion
- nearby movement
- other mechanically induced disturbances

This phase does **not** claim to detect:

- packet-level cyberattacks
- encryption failures
- data exfiltration
- malware
- routing attacks
- landing-station network attacks
- optical-layer cyber intrusion

Those belong to separate sensing/data layers and should not be conflated
with DAS physical-event detection.

---

## Primary References

1. Tomasov, A. et al. (2025). *Comprehensive Dataset for Event
   Classification Using Distributed Acoustic Sensing (DAS) Systems*.
   Scientific Data, 12, 793.
   DOI: 10.1038/s41597-025-05088-4

2. Duan, J., Chen, J., & He, Z. (2025). *DAS-MAE: A self-supervised
   pre-training framework for universal and high-performance
   representation learning of distributed fiber-optic acoustic sensing*.
   arXiv:2506.04552.

3. Ding, Q., Shen, Z., Zhu, W., et al. (2025). *DASFormer:
   self-supervised pretraining for earthquake monitoring*.
   Visual Intelligence, 3, 14.
   DOI: 10.1007/s44267-025-00085-y

4. Ramirez-Torres, E. E., et al. (2026). *A Distributed Acoustic Sensing
   Dataset for Vessel Detection and Localization in Submarine Cable
   Protection*. arXiv:2607.28306.

5. Spica, Z. et al. (2023). *PubDAS*.
   Seismological Research Letters.

6. SUBMERSE project datasets and associated submarine DAS research.

7. OOI Regional Cabled Array DAS datasets.

8. Trondheim submarine DAS / ship-tracking research.

9. Malik et al. (2025). Fiber-optic State-of-Polarization physical-threat
   detection research.
