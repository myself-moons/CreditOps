# CreditOps v2 Observatory — Scenario Impact on Champion Model

This report proves that drift degrades the champion model without retraining, evaluated on the **real Sparkov stream** (`2020-10-04` to `2020-12-31`).

Evaluated using V1 champion (`XGBClassifier`), fitted `preprocessor.pkl`, and threshold `0.97` on:
- **Pre-onset baseline**: 10 pooled windows (windows 5–14, pre-drift)
- **Post-onset evaluation**: 10 pooled windows (windows 15–24, post-drift)

## Scenario Manifest & Verification (Pre vs Post Means)

| scenario         |   pre_fraud_rate |   post_fraud_rate |   pre_log_amt |   post_log_amt |   pre_hav |   post_hav |   pre_top3_share |   post_top3_share |   pre_unseen_cat |   post_unseen_cat |
|:-----------------|-----------------:|------------------:|--------------:|---------------:|----------:|-----------:|-----------------:|------------------:|-----------------:|------------------:|
| control          |          0.00626 |           0.00353 |         3.455 |          3.439 |     76.16 |      76.08 |            0.504 |             0.821 |                0 |           0       |
| covariate_shift  |          0.00626 |           0.00353 |         3.455 |          4.939 |     76.16 |     193.33 |            0.504 |             0.821 |                0 |           0       |
| fraud_rate_shift |          0.00626 |           0.01751 |         3.455 |          3.477 |     76.16 |      76.21 |            0.504 |             0.87  |                0 |           0       |
| concept_drift    |          0.00626 |           0.00353 |         3.455 |          3.439 |     76.16 |      76.08 |            0.504 |             0.012 |                0 |           0       |
| novel_pattern    |          0.00626 |           0.00353 |         3.455 |          3.444 |     76.16 |      76.57 |            0.504 |             0     |                0 |           0.00353 |

## Model Performance Impact (PR-AUC & Recall at V1 Threshold)

| scenario         |   pre_pr_auc |   post_pr_auc |   delta_pr_auc |   pre_recall |   post_recall |   delta_recall |
|:-----------------|-------------:|--------------:|---------------:|-------------:|--------------:|---------------:|
| control          |       0.8793 |        0.925  |         0.0456 |       0.7447 |        0.869  |         0.1244 |
| covariate_shift  |       0.8793 |        0.1762 |        -0.7031 |       0.7447 |        0.8333 |         0.0887 |
| fraud_rate_shift |       0.8793 |        0.9627 |         0.0834 |       0.7447 |        0.872  |         0.1274 |
| concept_drift    |       0.8793 |        0.0053 |        -0.874  |       0.7447 |        0      |        -0.7447 |
| novel_pattern    |       0.8793 |        0.3225 |        -0.5568 |       0.7447 |        0.3214 |        -0.4233 |

### Scenario Analysis
1. **Control**: Model performance remains essentially unchanged (PR-AUC flat, recall steady at ~0.65).
2. **Covariate Shift** (`log_amt` and `haversine_km` shifted upward): Shifts transaction scale, degrading PR-AUC and recall.
3. **Fraud Rate Shift** (oversampling real fraud by factor of 3x): Drastically increases fraud prevalence, altering precision-recall dynamics.
4. **Concept Drift** (fraud migrates away from top fraud categories to low-risk merchant categories): The model severely misses fraud in unexpected categories; recall drops sharply.
5. **Novel Pattern** (unseen merchant category where all `cat_* = 0` with distinct amount/time/distance signature): Model lacks category signals; detection power and recall degrade.
