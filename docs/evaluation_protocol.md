# CreditOps v2 — Evaluation Protocol

> **Status:** Pre-registered protocol for Phase 5 final evaluation. Written prior to final evaluation runs.

---

## 1. Overview & Objectives

This protocol establishes the rigorous, pre-registered experimental setup for evaluating adaptive retraining policies and governed deployment in the CreditOps Continuous MLOps Drift Observatory.

The evaluation answers three core operational questions:
1. Does adaptive monitoring detect distribution drift and novel fraud patterns faster and with fewer false alarms than periodic schedules or single-metric thresholds?
2. Does governed challenger validation (holdout gating + threshold recalibration + shadow rollback) protect production financial risk compared to naive deployment?
3. What is the net financial benefit (fraud loss avoided minus operational and retraining overhead) across realistic drift scenarios?

---

## 2. Metrics & Formal Definitions

Let $T$ denote the total stream windows, $t_{\text{onset}}$ the window index where drift begins, and $\mathcal{W}_t$ the transaction set for window $t$ with ground-truth binary labels $y \in \{0, 1\}$.

### 2.1 Detection Delay
The number of windows elapsed between drift onset and the first policy intervention (alert or retrain):
$$\text{Delay} = t_{\text{first\_action}} - t_{\text{onset}} \quad \text{for } t_{\text{first\_action}} \ge t_{\text{onset}}$$
If no action is taken after onset, delay is undefined ($\infty$). For the `control` scenario, delay is not applicable (`—`).

### 2.2 False Alarms
The number of policy actions ($\text{action} \in \{\text{alert}, \text{retrain}\}$) triggered in the absence of drift:
- For drift scenarios ($t_{\text{onset}} > 0$): $\sum_{t < t_{\text{onset}}} \mathbb{I}(\text{action}_t \neq \text{no\_action})$
- For `control` scenario: $\sum_{t=0}^{T-1} \mathbb{I}(\text{action}_t \neq \text{no\_action})$

### 2.3 Financial Cost Model
Operational cost in each window $t$ is computed using the business cost parameters:
$$\text{Cost}(t) = c_{\text{FN}} \cdot \text{FN}_t + c_{\text{FP}} \cdot \text{FP}_t + c_{\text{review}} \cdot \mathbb{I}(\text{action}_t = \text{alert}) + c_{\text{retrain}} \cdot \mathbb{I}(\text{action}_t = \text{retrain})$$
Where:
- $c_{\text{FN}} = \$500$ (cost of missed fraud)
- $c_{\text{FP}} = \$5$ (cost of legitimate transaction declined)
- $c_{\text{review}} = \$10$ (analyst manual review overhead per alert)
- $c_{\text{retrain}} = \$200$ (pipeline execution and compute overhead per retrain attempt)

### 2.4 Performance Loss Avoided
The reduction in fraud classification error costs achieved by a policy $\pi$ compared to the non-intervention baseline $\text{NeverRetrain}$:
$$\text{Loss Avoided} = \text{Cost}_{\text{FN+FP}}(\text{NeverRetrain}) - \text{Cost}_{\text{FN+FP}}(\pi)$$

### 2.5 Net Benefit
The net financial gain accounting for retraining expenses:
$$\text{Net Benefit} = \text{Loss Avoided} - \text{Total Retrain Cost}(\pi)$$

### 2.6 Retraining Cost Accounting
Retraining cost ($c_{\text{retrain}} = \$200$) is charged on **every retrain trigger**, regardless of whether the challenger model is promoted, rejected at the gate, or subsequently rolled back.

---

## 3. Baselines & Policies

Five policies are evaluated with all parameters frozen from development:

| Policy | Trigger Mechanism | Deployment Mode | Description |
|---|---|---|---|
| **NeverRetrain** | None | N/A | Static baseline; champion model never updated. |
| **FixedSchedule (naive)** | Periodic ($K=20$) | Naive | Retrains on a fixed calendar schedule regardless of drift. |
| **ThresholdPolicy (naive)** | PSI Data Drift $> 1.0$ | Naive | V1 baseline; single aggregate covariate drift trigger with cooldown ($C=5$). |
| **AdaptivePolicy (naive)** | Persistent signals + Expected Benefit | Naive | Multi-detector monitoring (drift + novelty + delayed PR-AUC) with persistence ($N=2$) and economic cost gate. Deploys challenger immediately. |
| **AdaptivePolicy (governed)** | Persistent signals + Expected Benefit | Governed | Same adaptive decision signals, but deployment undergoes holdout validation ($K_{\text{val}}=3$), FP budget threshold recalibration, promotion gate ($\text{margin}_{\text{recall}} = 0.02$), shadow scoring ($M=5$), and automatic rollback ($\text{margin}_{\text{cost}} = \$100$). |

---

## 4. Evaluation Setup (Phase 5)

### 4.1 Temporal Stream & Data Isolation
- **Stream Period:** 2020-10-04 to 2020-12-31 (held-out test slice).
- **Window Size:** 2 calendar days per window (89 total windows).
- **Label Delay:** 2 windows ($t_{\text{train}} \le t - 2$, strictly enforced with automated assertions).
- **Zero Future Leakage:** All preprocessors and base models were fit on training periods ($< 2020-06-21$) prior to stream start.

### 4.2 Scenarios
- `control`: No drift injected. Verifies false alarm suppression.
- `covariate_shift`: Additive shift applied to transaction amounts and distances ($M=1.5$).
- `fraud_rate_shift`: 3x surge in fraud volume through oversampling real fraud ($M=3.0$).
- `novel_pattern`: Synthetic high-amount novel merchant category attack ($M=1.0$).
- *Note on `concept_drift`:* Excluded from primary evaluation per protocol (verified by forced-retrain study showing unrecoverable PR-AUC recovery on Sparkov December test slice).

### 4.3 Evaluation Seeds & Schedule
- **Onset Window:** $t_{\text{onset}} = 30$ (distinguishing pre-drift baseline from post-onset behavior).
- **Seeds:** Seed 1 and Seed 2 (unseen evaluation seeds; no parameter tuning permitted).
- **Reporting:** Mean $\pm$ standard deviation across evaluation seeds for all 5 policies $\times$ 4 scenarios.

---

## 5. Acceptance Thresholds & Decision Criteria

1. **Control Safety:**
   - $\text{False Retrains}_{\text{control}} = 0$ for Adaptive policies.
   - $\text{False Alarms}_{\text{control}} \le 1$.
2. **Governance Value:**
   - $\text{Net Benefit}(\text{Adaptive Governed}) \ge \text{Net Benefit}(\text{Adaptive Naive})$ in scenarios where challengers risk negative transfer or lack sufficient validation sample.
   - $\text{Net Benefit}(\text{Adaptive Governed}) \ge -\text{Total Retrain Cost}$ relative to NeverRetrain (governance never causes catastrophic unmitigated degradation).
3. **Data Leakage & Holdout Rigor:**
   - Holdout windows ($K_{\text{val}}$) must be strictly disjoint from challenger training windows: $\mathcal{W}_{\text{holdout}} \cap \mathcal{W}_{\text{train}} = \emptyset$.
   - Max window index in training data $\le t - \text{label\_delay}$.
4. **Audit Trail Completeness:**
   - 100% of candidate registrations, promotion gates, rejections, and rollbacks must be persisted to the `LogStore` (`model_versions` and `audit_log`) with cryptographic data hashes and decision reasons.
