"""
main.py — CreditOps FastAPI application

⚠ SIMULATED DATA NOTICE: This API serves a model trained on Sparkov-simulated
credit-card transactions. No real cardholder data is used or stored.

Routes:
  GET  /               Landing page
  GET  /dashboard      MLflow experiment dashboard
  GET  /predict        Prediction UI
  GET  /dataset        Dataset overview page
  GET  /docs           Swagger API docs
  GET  /api/dashboard  JSON dashboard data
  GET  /api/runs       JSON MLflow run history
  GET  /api/monitor    JSON operational monitoring stats
  GET  /api/dataset    JSON dataset profile
  GET  /health         Health check
  POST /predict        Fraud prediction endpoint
"""

import json
import os
import pickle
import time
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from src.data_model import Transaction
    from src.features import build_features, feature_names
    from src.monitor import get_operational_stats
    from src.prediction_logger import log_prediction
    from src.retrain_trigger import check_distribution_drift
except ImportError:
    from data_model import Transaction
    from features import build_features, feature_names
    from monitor import get_operational_stats
    from prediction_logger import log_prediction
    from retrain_trigger import check_distribution_drift

app = FastAPI(
    title="CreditOps — Credit Card Fraud Detection",
    description=(
        "⚠ SIMULATED DATA: This API serves a model trained on Sparkov-simulated "
        "credit-card transactions (kartik2112/fraud-detection on Kaggle). "
        "No real cardholders are involved. "
        "Fraud prediction API with MLflow tracking, DVC pipeline, and operational monitoring."
    ),
    version="1.0.0",
)

BASE_DIR          = Path(__file__).resolve().parent.parent
MODEL_PATH        = BASE_DIR / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"
METRICS_PATH      = BASE_DIR / "metrics.json"
PROFILE_PATH      = BASE_DIR / "dataset_profile.json"
SPLIT_INFO_PATH   = BASE_DIR / "split_info.json"
SUBGROUP_PATH     = BASE_DIR / "subgroup_metrics.csv"
THRESHOLD_PATH    = BASE_DIR / ".decision_threshold"
TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR / 'mlruns'}")

# Load model and preprocessor at startup
with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

with PREPROCESSOR_PATH.open("rb") as f:
    preprocessor = pickle.load(f)

threshold = 0.5
if THRESHOLD_PATH.exists():
    try:
        threshold = float(THRESHOLD_PATH.read_text().strip())
    except Exception:
        pass
if threshold == 0.5 and METRICS_PATH.exists():
    try:
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        threshold = float(m.get("threshold", 0.5))
    except Exception:
        pass
SELECTED_MODEL = type(model).__name__


# ============================================================================ #
# HTML page routes                                                              #
# ============================================================================ #
@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "src" / "landing.html").read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return (BASE_DIR / "src" / "dashboard.html").read_text(encoding="utf-8")


@app.get("/predict", response_class=HTMLResponse)
def prediction_page():
    return (BASE_DIR / "src" / "predict.html").read_text(encoding="utf-8")


@app.get("/dataset", response_class=HTMLResponse)
def dataset_page():
    return (BASE_DIR / "src" / "dataset.html").read_text(encoding="utf-8")


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page():
    return (BASE_DIR / "src" / "monitor.html").read_text(encoding="utf-8")


@app.get("/logs", response_class=HTMLResponse)
def logs_page():
    return (BASE_DIR / "src" / "logs.html").read_text(encoding="utf-8")


# ============================================================================ #
# Health                                                                        #
# ============================================================================ #
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model":         SELECTED_MODEL,
        "threshold":     threshold,
        "data_provenance": "synthetic/simulated (Sparkov)",
    }


# ============================================================================ #
# Internal helpers                                                              #
# ============================================================================ #
def _mlflow_runs() -> list:
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        exp = mlflow.get_experiment_by_name("credit-fraud")
        if exp is None:
            return []
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
        )
        result = []
        for _, r in runs.iterrows():
            def clean(v):
                return None if pd.isna(v) else v
            result.append({
                "run_id":          clean(r.get("run_id")),
                "run_name":        clean(r.get("tags.mlflow.runName")),
                "run_stage":       clean(r.get("tags.run_stage")),
                "status":          clean(r.get("status")) or "UNKNOWN",
                "start_time":      r.get("start_time").isoformat() if pd.notna(r.get("start_time")) else None,
                "model_type":      clean(r.get("params.model_type")),
                "cv_mean_pr_auc":  clean(r.get("metrics.cv_mean_pr_auc")),
                "cv_std_pr_auc":   clean(r.get("metrics.cv_std_pr_auc")),
                "val_pr_auc":      clean(r.get("metrics.val_pr_auc")),
                "val_recall":      clean(r.get("metrics.val_recall")),
                "test_pr_auc":     clean(r.get("metrics.test_pr_auc")),
                "test_roc_auc":    clean(r.get("metrics.test_roc_auc")),
                "test_f1_score":   clean(r.get("metrics.test_f1_score")),
                "test_precision":  clean(r.get("metrics.test_precision")),
                "test_recall":     clean(r.get("metrics.test_recall")),
                "test_recall_at_fp_budget": clean(r.get("metrics.test_recall_at_fp_budget")),
            })
        return result
    except Exception:
        return []


# ============================================================================ #
# JSON API routes                                                               #
# ============================================================================ #
@app.get("/api/dashboard")
def dashboard_data():
    runs    = _mlflow_runs()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    best_run = max(
        (r for r in runs if isinstance(r.get("test_pr_auc"), (float, int))),
        key=lambda r: r["test_pr_auc"],
        default=None,
    )
    subgroup_rows = []
    if SUBGROUP_PATH.exists():
        import csv
        with SUBGROUP_PATH.open() as f:
            reader = csv.DictReader(f)
            subgroup_rows = list(reader)

    return {
        "data_provenance": "synthetic/simulated (Sparkov)",
        "project": {
            "name":        "CreditOps — Credit Card Fraud Detection",
            "description": (
                "A DVC-managed MLOps pipeline that trains and serves a credit card fraud "
                "classifier. Data: Sparkov simulation (Kaggle kartik2112/fraud-detection). "
                "No real cardholder data is used."
            ),
            "pipeline": [
                "Data Collection", "Data Preprocessing",
                "Model Training", "Evaluation", "Dataset Profile",
                "Stationarity Check",
            ],
        },
        "results": metrics,
        "tracking": {
            "experiment":     "credit-fraud",
            "selected_model": SELECTED_MODEL,
            "threshold":      threshold,
            "run_count":      len(runs),
            "best_run":       best_run,
        },
        "runs":             runs,
        "subgroup_metrics": subgroup_rows[:50],  # top 50 for dashboard
    }


@app.get("/api/runs")
def api_runs():
    return {
        "experiment":      "credit-fraud",
        "data_provenance": "synthetic/simulated (Sparkov)",
        "runs":            _mlflow_runs(),
    }


@app.get("/api/monitor")
def monitor_status():
    """
    Operational monitoring statistics, distribution drift check, and monitoring architecture.

    Layer A: Operational monitoring (real-time, zero ground truth needed).
      - Fraud detection rate shift vs training baseline
      - Probability distribution calibration
      - Feature stationarity (PSI & KS statistics)
      - API latency SLA (Mean & P95)

    Layer B: Model performance monitoring (delayed ground truth).
      - PR-AUC degradation (alert if < 0.10)
      - Recall drop at 5% FP budget (alert if < 0.60)
      - Automated retraining recommendations
    """
    stats = get_operational_stats()
    drift = check_distribution_drift()

    stationarity_summary = {}
    stationarity_file = BASE_DIR / "stationarity_report.json"
    if stationarity_file.exists():
        try:
            st_data = json.loads(stationarity_file.read_text(encoding="utf-8"))
            stationarity_summary = st_data.get("summary", {})
            stationarity_summary["conclusion"] = st_data.get("conclusion", "")
        except Exception:
            pass

    return {
        "data_provenance": "synthetic/simulated (Sparkov)",
        "operational":     stats,
        "drift_check":     drift,
        "stationarity":    stationarity_summary,
        "monitoring_architecture": {
            "layer_a_operational": {
                "name": "Layer A — Operational Drift & System Health",
                "ground_truth_required": False,
                "metrics_tracked": [
                    "Rolling Fraud Detection Rate vs 0.579% Baseline",
                    "Score Probability Distribution (Mean, Min, Max, StDev)",
                    "Population Stability Index (PSI) per feature (Threshold < 0.10)",
                    "Kolmogorov-Smirnov (KS) Two-Sample Test (Tolerance < 0.05)",
                    "Inference Latency SLA (Mean & P95 ms)"
                ],
                "status": "active"
            },
            "layer_b_performance": {
                "name": "Layer B — Model Performance & Retrain Governance",
                "ground_truth_required": True,
                "metrics_tracked": [
                    "PR-AUC (Precision-Recall Area Under Curve, Trigger: < 0.10)",
                    "Recall at 5% False-Positive Budget (Trigger: < 0.60)",
                    "Subgroup Parity Across Transaction Categories & Age Bands"
                ],
                "status": "ready_for_eval"
            },
            "retraining_triggers": {
                "pr_auc_threshold": 0.10,
                "recall_threshold": 0.60,
                "training_fraud_rate_baseline": 0.00579,
                "drift_tolerance": 0.003
            }
        },
        "note": (
            "PR-AUC, precision, recall, and F1 require ground-truth fraud labels. "
            "These become available when actual fraud outcomes are confirmed — "
            "typically days to weeks after the transaction. "
            "Any drift detected is compared against a Sparkov-simulation baseline."
        ),
    }


@app.get("/api/stationarity")
def stationarity_api():
    """Returns the full stationarity report (PSI and KS tests across all splits and months)."""
    st_file = BASE_DIR / "stationarity_report.json"
    if st_file.exists():
        data = json.loads(st_file.read_text(encoding="utf-8"))
        data.setdefault("data_provenance", "synthetic/simulated (Sparkov)")
        return data
    return {
        "data_provenance": "synthetic/simulated (Sparkov)",
        "error": "stationarity_report.json not found — run dvc repro Stationarity_Check",
    }


@app.get("/api/predictions/recent")
def recent_predictions(limit: int = 50):
    """Returns recent prediction records from the log for real-time monitoring."""
    log_file = BASE_DIR / "predictions.jsonl"
    if not log_file.exists():
        return {"total": 0, "records": [], "data_provenance": "synthetic/simulated (Sparkov)"}
    records = []
    with log_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return {
        "total": len(records),
        "records": records[-limit:][::-1],
        "data_provenance": "synthetic/simulated (Sparkov)",
    }


@app.get("/api/dataset")
def dataset_api():
    """JSON endpoint backing the /dataset page. Reads dataset_profile.json."""
    if PROFILE_PATH.exists():
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        data.setdefault("data_provenance", "synthetic/simulated (Sparkov)")
        return data
    return {
        "data_provenance": "synthetic/simulated (Sparkov)",
        "error":           "dataset_profile.json not found — run dvc repro Dataset_Profile",
    }


# ============================================================================ #
# Prediction endpoint                                                           #
# ============================================================================ #
@app.post("/predict")
def model_predict(payload: Transaction):
    """
    Predict whether a credit card transaction is fraudulent.

    Returns:
      - prediction: 0 (legitimate) or 1 (fraud)
      - fraud_probability: model confidence for the fraud class
      - decision: "fraud" or "legitimate"
      - threshold: decision boundary used

    ⚠ SIMULATED DATA: Trained on Sparkov simulation, not real transactions.
    """
    start_ms = time.time() * 1000

    # Build raw row dict matching what features.py expects
    raw_row = {
        "trans_date_trans_time": payload.trans_date_trans_time,
        "amt":                   payload.amt,
        "category":              payload.category,
        "gender":                payload.gender,
        "dob":                   payload.dob,
        "lat":                   payload.lat,
        "long":                  payload.long,
        "city_pop":              payload.city_pop,
        "merch_lat":             payload.merch_lat,
        "merch_long":            payload.merch_long,
    }

    try:
        raw_df       = pd.DataFrame([raw_row])
        features_df  = build_features(raw_df)
        features_enc = preprocessor.transform(features_df)
        fraud_prob   = float(model.predict_proba(features_enc)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Inference processing failed: {str(exc)}"
        )

    predicted    = int(fraud_prob >= threshold)
    decision     = "fraud" if predicted == 1 else "legitimate"
    latency_ms   = time.time() * 1000 - start_ms

    # Log prediction (derived features + transaction metadata, no PII)
    derived = features_df.iloc[0].to_dict()
    derived["amt"] = round(float(payload.amt), 2)
    derived["category"] = str(payload.category)

    log_prediction(
        prediction=predicted,
        probability=fraud_prob,
        decision=decision,
        model_version=SELECTED_MODEL,
        latency_ms=latency_ms,
        derived_features=derived,
    )

    return {
        "prediction":        predicted,
        "fraud":             predicted == 1,
        "decision":          decision,
        "fraud_probability": round(fraud_prob, 4),
        "threshold":         round(threshold, 4),
        "model":             SELECTED_MODEL,
        "latency_ms":        round(latency_ms, 2),
        "data_provenance":   "synthetic/simulated (Sparkov)",
    }
