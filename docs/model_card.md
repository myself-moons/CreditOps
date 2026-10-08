# Model Card: CreditOps Fraud Detection Models (V1 Champion & V2 Governed Adaptive)

## Model Details
- **Organization**: CreditOps Architecture Team
- **Model Dates**: October 2026
- **Model Versions**: 
  - `v1.0` (Champion Base): Static XGBoost model trained on Sparkov-simulated transaction stream (train slice: 2019-01-01 to 2020-07-31).
  - `v2.x` (Governed Retrained Champions): Continual learning champions deployed via the CreditOps Observatory governance pipeline.
- **Model Type**: Gradient Boosted Decision Trees (`XGBClassifier`), `scale_pos_weight=1.0`, decision threshold calibrated at 0.97 for 0.5% FPR target.

## Intended Use
- **Primary Use Case**: Real-time fraud detection in cardholder transactional streams.
- **Out-of-Scope Use Cases**: Real human cardholder profiling or credit scoring without customer consent.
- **Synthetic Data Notice**: Trained and benchmarked entirely on Sparkov synthetic transaction streams (`kartik2112/fraud-detection`). No real PII is stored or inferred.

## Factors & Performance Metrics
- **Primary Metric**: PR-AUC (Precision-Recall Area Under Curve), preferred due to extreme class imbalance (~0.5% fraud rate).
- **Secondary Metrics**: Recall@Fixed Threshold ($T=0.97$), False Positive Rate (FPR $\le 0.5\%$), Total Financial Cost ($C_{FN}=\$500$, $C_{FP}=\$5$).
- **Benchmark Performance (Test Slice)**:
  - Base Test PR-AUC: $0.852$ (clean temporal validation)
  - Stationary Stream PR-AUC: $0.6909 \pm 0.0000$
  - Novel Pattern Adaptation PR-AUC: $0.6920$ (Governed Adaptive) vs $0.3050$ (NeverRetrain static champion failure).

## Governance & Safety Controls
- **Validation Gating**: $K_{val}=3$ disjoint temporal windows buffer. Challengers must beat current champion recall with $\le 5\%$ cost degradation margin.
- **Shadow Monitoring**: Promoted champions run in shadow mode alongside incumbent champions for $M=3$ windows ($6$ calendar days).
- **Automated Rollback**: If shadow window cost exceeds incumbent by $> \$100$, immediate automated fallback to previous champion is enacted.
- **Privacy & Security**: PII stripped and hashed (SHA-256) at ingestion; RBAC enforcement on rollback (`risk_owner`) and replay (`ml_engineer`).
