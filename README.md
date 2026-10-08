# CreditOps

## Credit Card Fraud Detection MLOps System

A reproducible, end-to-end MLOps pipeline that trains and serves a credit card fraud classifier.

**Maintainer:** Manthan Waghela / myself-moons  
**Live Deployment:** [https://creditops.onrender.com](https://creditops.onrender.com)

> ⚠️ **SIMULATED DATA** — This project uses the Sparkov-generated Credit Card Transactions Fraud
> Detection Dataset (Kaggle: kartik2112/fraud-detection, License: CC0 1.0).
> No real cardholders, card numbers, or merchants are involved.
> See [docs/data_disclosure.md](docs/data_disclosure.md) for the full disclosure.

```
Credit_Data/ (fraudTrain.csv + fraudTest.csv)
        ↓  DVC
Data Collection (concat + temporal 70/15/15 split)
        ↓
features.py (single source of truth for train + API)
        ↓
Data Preprocessing (sklearn ColumnTransformer → preprocessor.pkl)
        ↓
Model Training (LR / RF / XGBoost → CV selection by PR-AUC → model.pkl)
        ↓  MLflow (experiment: credit-fraud)
Model Evaluation (PR-AUC · Subgroup metrics → metrics.json)
        ↓
Dataset Profile (dataset_profile.json)
        ↓
Stationarity Check (PSI / KS / monthly PR-AUC → stationarity_report.json)
        ↓
FastAPI Serving (POST /predict · GET /dataset · GET /api/monitor)
        ↓
Prediction Logging (predictions.jsonl — derived features only, no PII)
        ↓
Operational Monitoring (/api/monitor)
        ↓
Retraining Trigger (PR-AUC + recall thresholds in params.yaml)
```

---

## Metrics (measured from real simulated data)

> All numbers below are extracted directly from an actual training run (`metrics.json` and MLflow tracking run).

| Model | Test PR-AUC | Test ROC-AUC | Test Recall | Test F1 | Recall@5%FPR | Test Accuracy | Decision Threshold |
|---|---|---|---|---|---|---|---|
| **XGBoost (Champion)** | **0.8722** | **0.9977** | **79.33%** | **0.8236** | **0.11%** | **99.89%** | **0.97** |

*Evaluated on held-out temporal test split (277,860 transactions, 924 fraud cases, 0.3325% fraud rate).*

### Candidate Model Cross-Validation (3-Fold CV PR-AUC)

| Model | 3-Fold CV Mean PR-AUC | Std Dev | Role |
|---|---|---|---|
| **XGBoost** | **0.9553** | ±0.0020 | **Champion** — selected for production serving, gradient boosted trees with `scale_pos_weight` |
| **Random Forest** | **0.8341** | ±0.0055 | Benchmark ensemble (100 estimators, max depth 6) |
| **Logistic Regression** | **0.3485** | ±0.0032 | Regularized linear baseline |

**Primary metric: PR-AUC** — at ~0.5% fraud rate, accuracy and ROC-AUC are misleading.
PR-AUC measures performance on the minority (fraud) class across all thresholds.

---

## Problem Statement

Credit card fraud detection is a highly imbalanced classification problem (~0.5% fraud rate).
This project builds a complete MLOps pipeline that:

- Trains multiple classifiers on **simulated** Sparkov transaction data
- Selects the best model by **PR-AUC** (not accuracy or ROC-AUC)
- Performs **temporal splitting** to prevent temporal leakage
- Exposes a **REST API** for real-time fraud scoring
- Logs predictions (derived features only, no PII) for operational monitoring
- Measures data **stationarity** and reports it honestly

---

## Dataset

**Source:** Kaggle "Credit Card Transactions Fraud Detection Dataset" (kartik2112/fraud-detection)  
**Generator:** Sparkov (namebrandon/Sparkov_Data_Generation)  
**License:** CC0 1.0 Universal (Public Domain Dedication)  
**⚠️ This is simulated data. No real cardholders or transactions are involved.**

| File | Rows | Date Range |
|---|---|---|
| `fraudTrain.csv` | 1,296,675 | Jan 2019 – Jun 2020 |
| `fraudTest.csv` | 555,719 | Jun 2020 – Dec 2020 |
| **Total** | **1,852,394** | **Jan 2019 – Dec 2020** |

Overall fraud rate: ~0.5% (highly imbalanced).  
Raw data is tracked with DVC (`Credit_Data/` is in `.gitignore`).

### Temporal Split

| Split | Fraction | Rows | Period | Fraud Rate | Fraud Count |
|---|---|---|---|---|---|
| Train | 70% | 1,296,675 | 2019-01-01 → 2020-06-21 | 0.5789% | 7,506 |
| Validation | 15% | 277,859 | 2020-06-21 → 2020-10-03 | 0.4394% | 1,221 |
| Test | 15% | 277,860 | 2020-10-03 → 2020-12-31 | 0.3325% | 924 |

Random splitting is not used — see [Why Temporal Split](#why-temporal-split).

---

## Architecture

```
src/
  dataset_adapter.py      DatasetAdapter class (single data access point)
  features.py             Single source of truth for feature engineering
  data_collection.py      Load + temporal split → data/raw/
  data_preprocessing.py   ColumnTransformer → preprocessor.pkl
  model_training.py       CV selection by PR-AUC → model.pkl + .decision_threshold
  model_evaluation.py     Full metrics + subgroup tables → metrics.json
  dataset_profile.py      Dataset statistics → dataset_profile.json
  stationarity_check.py   PSI/KS + monthly PR-AUC → stationarity_report.json
  data_model.py           Transaction Pydantic model (API validation)
  main.py                 FastAPI application (/, /predict, /dataset, /api/*)
  prediction_logger.py    Append predictions to predictions.jsonl (no PII)
  monitor.py              Operational + performance monitoring
  retrain_trigger.py      PR-AUC + recall threshold-based retraining decision
  landing.html            Project overview page
  dashboard.html          MLflow run comparison dashboard
  predict.html            Transaction fraud prediction form
  dataset.html            Dataset overview with charts and data dictionary
  monitor.html            Operational and drift monitoring dashboard
  logs.html               Real-time transaction prediction audit logs
tests/
  test_api.py             Full test suite (API, validation, temporal split, features)
docs/
  data_disclosure.md      Full synthetic data statement and license
  limitations.md          Honest limitations and injected-drift rules
  known_issues.md         Stubbed or deferred features
```

---

## Feature Engineering (`src/features.py`)

All feature derivation lives in a single function `build_features()`, called by both
training and the API to guarantee identical transformations:

| Feature | Derivation |
|---|---|
| `hour` | Hour of transaction (0–23) |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `log_amt` | log(amt + ε) — right-skewed distribution |
| `cat_*` (14) | One-hot of 14 transaction categories |
| `gender_M` | 1 if Male, 0 if Female |
| `age_at_txn` | (transaction date − DOB) in years |
| `haversine_km` | Great-circle distance cardholder ↔ merchant |
| `log_city_pop` | log(city_pop + ε) |

**Dropped (privacy / leakage / cardinality):** `cc_num`, `first`, `last`, `street`, `trans_num`, `zip`, `dob` (after age), `merchant`, `city`, `state`, `job`, `unix_time`, raw lat/long (after haversine).

---

## ML Pipeline

### 1. Data Collection
- Concatenates train + test CSVs, parses timestamps, drops duplicates
- Validates schema against expected columns
- Temporal split: 70% train / 15% val / 15% test (by time order, no randomness)
- Outputs `data/raw/{train,val,test}.csv` and `split_info.json`

### 2. Preprocessing
- `features.py` converts raw rows to feature vectors
- `sklearn ColumnTransformer`: StandardScaler on continuous features, passthrough on binary/one-hot
- **Fitted on training data only** — no val/test data touches the fit step
- Saves `preprocessor.pkl`

### 3. Model Training
- Candidates: Logistic Regression, Random Forest, XGBoost (+ LightGBM if installed)
- **Imbalance handling:** `class_weight="balanced"` (LR/RF), `scale_pos_weight` (XGBoost)
- **Majority downsampling** for training speed (30% of non-fraud kept; configurable in `params.yaml`)
- **CV selection** on training data by 3-fold cross-val PR-AUC (not ROC-AUC)
- Winner retrained on full training split
- **Threshold tuning** on validation set (max F1), saved to `.decision_threshold`
- Final eval on untouched test split
- All runs logged to MLflow experiment `credit-fraud`; champion registered in Model Registry

### 4. Evaluation
- Overall: PR-AUC, ROC-AUC, precision, recall, F1, recall@5%FPR
- Subgroup: by category, amount bucket, age band, distance bucket, hour of day
- Outputs `metrics.json` and `subgroup_metrics.csv`

---

## Why Temporal Split?

A random split allows the model to train on future transactions — for example, new merchants that
appear later in time could leak temporal patterns. Fraud behaviour is also time-dependent
(seasonality, new fraud patterns emerge over time). Temporal splitting preserves natural
temporal ordering and matches real deployment conditions where the model is always predicting
on future data.

## Why PR-AUC?

At ~0.5% fraud rate:
- A model predicting "legitimate" for every transaction achieves 99.5% accuracy
- ROC-AUC can be inflated even for poor models because true negatives (legitimate transactions) dominate
- PR-AUC focuses on how well the model ranks the rare positives (fraud) relative to negatives
- It is the correct metric when the cost of missing a fraud (false negative) is high

---

## DVC Pipeline

```bash
# Reproduce the full pipeline from raw data
dvc repro

# Run specific stage only
dvc repro Data_Collection
dvc repro Model_Training
```

Stages: `Data_Collection → Data_Preprocessing → Model_Training → Evaluation → Dataset_Profile → Stationarity_Check`

---

## Running the API

```bash
# After dvc repro completes:
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or with venv:
.venv/Scripts/uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Pages: `/` (landing) · `/predict` · `/dashboard` · `/dataset` · `/monitor` · `/logs` · `/docs` (Swagger)

API endpoints:
- `GET  /health`         — model & system health status
- `POST /predict`        — fraud prediction
- `GET  /api/monitor`    — operational monitoring stats & PSI metrics
- `GET  /api/dataset`    — dataset profile JSON
- `GET  /api/dashboard`  — MLflow run history + metrics
- `GET  /api/runs`       — MLflow runs only
- `GET  /api/logs`       — recent transaction prediction logs

### Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2020-06-15 14:32:00",
    "category": "travel",
    "amt": 850.00,
    "gender": "M",
    "dob": "1970-05-15",
    "lat": 36.1,
    "long": -115.2,
    "city_pop": 50000,
    "merch_lat": 34.1,
    "merch_long": -118.3
  }'
```

---

## Tests

```bash
pytest -q tests/
```

Tests cover: API routes (200/422), prediction structure, temporal split leakage,
preprocessor fit-on-train-only, features.py consistency (batch = API path),
synthetic data notice in all pages and JSON endpoints.

---

## Monitoring

The system provides two monitoring layers:

**Layer A (no ground truth needed):** Prediction count, fraud detection rate, probability distribution, latency, model version.

**Layer B (requires ground truth):** PR-AUC, recall, precision, F1 against actual fraud outcomes. In real deployment, fraud labels become available days/weeks after the transaction.

Distribution drift is detected by comparing the recent fraud detection rate against the training baseline (configurable in `params.yaml`).

---

## V2-Readiness

The following design patterns are in place for the capstone extension:

- `DatasetAdapter` class (single point of data access — swap dataset by changing one class)
- `time_windows()` method for windowed monitoring/stationarity
- All thresholds in `params.yaml`, none hardcoded
- `src/` modular: each concern is a separate module
- MLflow Model Registry with `champion` alias

**Any capstone drift scenarios must be:** injected, seeded, config-driven, and labelled "injected"
with a known onset window. See [docs/limitations.md](docs/limitations.md).

---

## Stationarity

The Sparkov simulation is nearly stationary (measured mean PSI: **0.0173** across all features and months, well below the 0.10 threshold).
This is measured and reported honestly in `stationarity_report.json` after `dvc repro Stationarity_Check`.
Any variation reported is simulation-internal, not real-world concept drift.

---

---

## Capstone Extension: CreditOps v2 Continuous Drift Observatory & Governance Platform

CreditOps v2 extends the static production system with a continuous MLOps evaluation platform. It simulates real-time transaction streaming, tests multi-detector drift alarms, evaluates adaptive retraining policies, and enforces strict governance gates (holdout validation, threshold recalibration, shadow validation, and automated rollback).

### Key Empirical Results (Frozen `eval-v1` Protocol across 48 Runs)

| Scenario | Governed Adaptive Net Benefit ($) | Fixed Schedule Churn ($) | Governance Protection |
| :--- | :---: | :---: | :--- |
| **`control`** (Stationary) | **$0.00** (0 false retrains) | **-$113,620.00** (5 unnecessary retrains) | **AC-1 Passed**: 0 false alarms, 0 churn |
| **`covariate_shift`** (Spending shift) | **+$39,785.00** (8 retrains) | **-$51,227.50** (12 windows delay) | Beats naive adaptive by +$34k; beats fixed schedule by +$90k |
| **`fraud_rate_shift`** (3x surge) | **-$200.00** (Candidate rejected at gate) | **-$468,592.50** (Severe calibration loss) | **AC-2 Passed**: Gate blocks uncalibrated model, saving $370k+ |
| **`novel_pattern`** (Zero-day attack) | **+$156,997.50** (0 windows delay) | **+$96,442.50** (12 windows delay) | Unseen category detector alarms on day 0 |

*All results verified under pre-registered protocol [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) and summarized in [`docs/results_summary.md`](docs/results_summary.md).*

### Architectural Differences: v1 vs v2

| Dimension | CreditOps v1 (Production Base) | CreditOps v2 (Continuous Observatory) |
| :--- | :--- | :--- |
| **Inference Mode** | Real-time REST API scoring (`/predict`) | Streaming 2-day temporal window replay (`/observatory/api/live`) |
| **Drift Detection** | Offline monthly PSI batch job | Real-time multi-signal detectors (Data Drift PSI/KS, Novelty, Fraud Score PSI, Delayed PR-AUC) |
| **Retraining Decision** | Manual review threshold trigger | Cost-aware economic gating ($H=10$ windows expected horizon vs $c_{\text{retrain}}=\$200$) |
| **Deployment Mode** | Direct model promotion | Governed deployment: holdout buffer ($K_{\text{val}}=3$), FP budget recalibration, shadow scoring ($M=5$), auto-rollback |
| **Audit & Storage** | Append-only `predictions.jsonl` | Dual-backend `LogStore` (Firebase RTDB + SQLite) with cryptographic provenance & SHA-256 token hashing |

### Core Observatory API & Endpoints
- **Web Console UI**: `http://localhost:8000/observatory` (and linked from `/dashboard`)
- **Summary & Criteria**: `GET /observatory/api/summary`
- **Window Traces**: `GET /observatory/api/trace/{scenario}/{policy}`
- **Model Registry & Audit Log**: `GET /observatory/api/models`, `GET /observatory/api/audit`
- **Live Stream Telemetry**: `GET /observatory/api/live`
- **Authorized Rollbacks (RBAC)**: `POST /observatory/api/rollback` (requires `risk_owner` token)
- **Authorized Replay (RBAC)**: `POST /observatory/api/replay` (requires `ml_engineer` token)

### Quick Start (Demo Mode)
```bash
# Seed demo run into database (Firebase RTDB or SQLite) from real evaluation traces
python scripts/seed_demo.py

# Launch FastAPI application
uvicorn src.main:app --host 127.0.0.1 --port 8000

# Open in browser:
# http://127.0.0.1:8000/observatory
```

### Reproducing the Complete 48-Run Evaluation Suite
```bash
# Run frozen evaluation suite (supports automatic resumption if interrupted)
python experiments/run_all.py

# Results and traces generated in:
# - results/results.csv (Summary table)
# - results/evaluation_report.json (Bitwise verification report)
# - results/runs/*.json (Window-by-window detailed traces)
```

---

## Documentation Index

- [CAPSTONE_PLAN.md](docs/CAPSTONE_PLAN.md) — Capstone development roadmap and phase completion log.
- [results_summary.md](docs/results_summary.md) — Phase 5 final evaluation results, tables, and hypothesis tests.
- [evaluation_protocol.md](docs/evaluation_protocol.md) — Pre-registered experimental setup, cost models, and acceptance thresholds.
- [model_card.md](docs/model_card.md) — Official model card for V1 Champion and V2 Governed models.
- [limitations.md](docs/limitations.md) — System boundaries, generative artifacts, and label delay assumptions.
- [known_issues.md](docs/known_issues.md) — Operational notes, credential masking, and deployment details.
- [data_disclosure.md](docs/data_disclosure.md) — Synthetic data provenance and license statements.
- [observatory_data_dictionary.md](docs/observatory_data_dictionary.md) — Schema definitions for streaming windows, telemetry, and traces.
