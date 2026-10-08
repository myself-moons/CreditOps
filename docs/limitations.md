# CreditOps v2 — System Limitations & Boundary Conditions

This document details the operational boundaries, empirical limitations, and scope constraints of the CreditOps fraud detection and retraining platform.

---

## 1. Synthetic Data & Sparkov Generative Constraints
- **Generative Process Artifacts**: The evaluation stream is built on the Sparkov synthetic dataset (`kartik2112/fraud-detection`). While it reproduces realistic diurnal spending cycles, geographical distances, and Merchant Category Codes (MCCs), synthetic generators have inherent statistical regularities not found in adversarial fraud rings.
- **Extreme Class Imbalance**: The baseline fraud prevalence is approximately $0.5\%$. In windows with fewer than $N=5$ fraud cases, empirical PR-AUC estimates exhibit higher variance.

---

## 2. Pre-Registered Exclusion of `concept_drift`
- **Protocol Pre-Registration**: As documented in `docs/evaluation_protocol.md` under frozen tag `eval-v1`, the `concept_drift` scenario (fraud migration across existing categories without covariate drift) was formally excluded from the final Phase 5 evaluation suite.
- **Empirical Rationale**: In the development setup (onset 15), stream-only retraining on 2-day windows ($K_{retrain}=10$) proved mathematically insufficient to invert the concept drift boundary without access to millions of non-fraud historical transactions outside the stream window. Including it would obscure the comparative evaluation of operational governance mechanisms. Full development setup metrics for `concept_drift` are archived in `docs/detector_sanity.md` and `docs/scenario_impact.md`.

---

## 3. Temporal Windowing & Operational Delays
- **Label Feedback Lag ($L=2$ windows)**: Ground truth labels are assumed to arrive with a minimum 2-window delay (4 days). While this is realistic for chargeback initiation (which can take 7 to 30 days in card networks), real-world fraud programs face even longer tail-delays.
- **Fixed Window Size ($\Delta t = 2$ days)**: All evaluation runs use fixed 2-day temporal window intervals. Sub-daily streaming micro-shifts or seasonal annual shifts were not evaluated.

---

## 4. Model Architecture & Scope
- **Classifier Class**: The governance framework was benchmarked on Gradient Boosted Decision Trees (`XGBClassifier`). Deep learning sequential models (e.g., Temporal Graph Networks, LSTMs, Transformers) and online continuous learners (e.g., River, Hoeffding Trees) were outside the capstone project scope.
- **Hyperparameter Stability**: Tree depth, learning rate, and feature engineering specifications were frozen under `eval-v1` to eliminate confounding variables during governance evaluation.
