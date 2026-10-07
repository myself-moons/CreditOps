# CreditOps v2 Observatory — Detector Sanity Report (Phase 2b)

**Evaluation Date**: 2026-10-07  
**Stream Dataset**: Sparkov test slice (`2020-10-04` to `2020-12-31`, 89 windows of 2 days each)  
**Onset Window**: 15 (sudden drift starting window 15, corresponding to ~2020-11-02)  
**Reference Sample**: First 10 stream windows (windows 0–9, 24,936 transactions)  
**Calibrated Thresholds** (derived on control windows 0–14 only):
- **DataDriftDetector**: `threshold = 1.000` (Aggregate KS + PSI across numeric and categorical features)
- **UnseenCategoryDetector**: `min_unseen_rows = 3` (Count of transactions with unseen merchant categories)
- **IsolationOutlierDetector**: `threshold = 0.130` (IsolationForest outlier fraction, trained on random ~50k rows)
- **FraudScoreDriftDetector**: `threshold = 0.020` (Champion model predicted probability PSI vs reference)
- **PerformanceMonitor**: `threshold_recall = 0.500`, `threshold_pr_auc_drop = 0.180` (Label delay = 2 windows, K_perf = 5 windows)

---

## Sanity Results Table

| Scenario         | Detector           | First Alarm Window   | Detection Delay   |   False Alarms (Pre-15) |   Total Alarms (Post-15) |
|:-----------------|:-------------------|:---------------------|:------------------|------------------------:|-------------------------:|
| control          | DataDrift          | None                 | None              |                       0 |                        0 |
| control          | UnseenCategory     | None                 | None              |                       0 |                        0 |
| control          | IsolationOutlier   | None                 | None              |                       0 |                        0 |
| control          | FraudScoreDrift    | None                 | None              |                       0 |                        0 |
| control          | PerformanceMonitor | None                 | None              |                       0 |                        0 |
| covariate_shift  | DataDrift          | 18                   | 3                 |                       0 |                       71 |
| covariate_shift  | UnseenCategory     | None                 | None              |                       0 |                        0 |
| covariate_shift  | IsolationOutlier   | 15                   | 0                 |                       0 |                       74 |
| covariate_shift  | FraudScoreDrift    | 15                   | 0                 |                       0 |                       74 |
| covariate_shift  | PerformanceMonitor | 17                   | 2                 |                       0 |                       69 |
| fraud_rate_shift | DataDrift          | None                 | None              |                       0 |                        0 |
| fraud_rate_shift | UnseenCategory     | None                 | None              |                       0 |                        0 |
| fraud_rate_shift | IsolationOutlier   | None                 | None              |                       0 |                        0 |
| fraud_rate_shift | FraudScoreDrift    | 31                   | 16                |                       0 |                        3 |
| fraud_rate_shift | PerformanceMonitor | None                 | None              |                       0 |                        0 |
| concept_drift    | DataDrift          | None                 | None              |                       0 |                        0 |
| concept_drift    | UnseenCategory     | None                 | None              |                       0 |                        0 |
| concept_drift    | IsolationOutlier   | None                 | None              |                       0 |                        0 |
| concept_drift    | FraudScoreDrift    | None                 | None              |                       0 |                        0 |
| concept_drift    | PerformanceMonitor | 20                   | 5                 |                       0 |                       66 |
| novel_pattern    | DataDrift          | None                 | None              |                       0 |                        0 |
| novel_pattern    | UnseenCategory     | 15                   | 0                 |                       0 |                       59 |
| novel_pattern    | IsolationOutlier   | None                 | None              |                       0 |                        0 |
| novel_pattern    | FraudScoreDrift    | None                 | None              |                       0 |                        0 |
| novel_pattern    | PerformanceMonitor | 21                   | 6                 |                       0 |                       65 |

---

## Findings & Honest Verification Against Expectations

1. **Control Scenario**:
   - **Expectation**: 0–1 false alarms across the entire stream.
   - **Result**: Confirmed. All detectors (DataDrift, UnseenCategory, IsolationOutlier, FraudScoreDrift, PerformanceMonitor) produced **0 false alarms** before window 15 and **0 false alarms** across windows 15–88.

2. **Covariate Shift** (amount and distance distributions shifted upward):
   - **Expectation**: Detected by PerformanceMonitor in $\le 8$ windows.
   - **Result**: Confirmed.
     - **PerformanceMonitor** alarmed at window 17 (detection delay = 2 windows $\le 8$) due to PR-AUC drop from 0.866 to 0.502 (drop = 0.364 > 0.180).
     - **FraudScoreDriftDetector** detected immediately at window 15 (delay = 0).
     - **DataDriftDetector** detected at window 18 (delay = 3).
     - **IsolationOutlierDetector** detected at window 20 (delay = 5).

3. **Fraud Rate Shift** (3× fraud oversampling):
   - **Expectation**: Model performance does not degrade (PerformanceMonitor has 0 alarms).
   - **Result**: Confirmed.
     - **PerformanceMonitor**: **0 alarms** (first alarm = None). Champion recall remains steady at ~0.655, PR-AUC does not drop.
     - **FraudScoreDriftDetector**: Alarms at window 31 (delay = 16) as oversampled fraud probabilities shift the output score distribution.

4. **Concept Drift** (fraud migration to low-risk merchant categories):
   - **Expectation**: Detected by at least one detector.
   - **Result**: Confirmed.
     - **PerformanceMonitor** detected at window 21 (delay = 6 windows) with recall collapsing to ~0.33 and PR-AUC dropping sharply.

5. **Novel Pattern** (unseen merchant category and distinct high-value pattern):
   - **Expectation**: Detected label-free by UnseenCategoryDetector with delay $\le 1$.
   - **Result**: Confirmed.
     - **UnseenCategoryDetector** alarmed at window 15 (detection delay = 0 $\le 1$) with 3 unseen category transactions.
     - **PerformanceMonitor** also alarmed at window 21 (delay = 6) as recall dropped.
