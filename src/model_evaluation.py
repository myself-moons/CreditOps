"""
model_evaluation.py — CreditOps

DVC Stage: Evaluation
-----------------------
Loads model.pkl + preprocessor.pkl, evaluates on the test split,
produces:
  - metrics.json          (overall + subgroup metrics)
  - subgroup_metrics.csv  (per category, amount bucket, age band,
                            distance bucket, hour of day)

PRIMARY METRIC: PR-AUC (average precision).
All numbers in this file come from the real (simulated) test split.

SIMULATED DATA NOTICE: Source data is Sparkov simulation.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import build_features, feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "is_fraud"


def compute_metrics(y_true, y_prob, threshold: float = 0.5, fp_budget: float = 0.05) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    n_neg  = int((y_true == 0).sum())
    fp_budget_count = int(n_neg * fp_budget)

    recall_at_budget = 0.0
    for t in sorted(np.unique(y_prob), reverse=True):
        y_p = (y_prob >= t).astype(int)
        fp_count = int(((y_p == 1) & (y_true == 0)).sum())
        if fp_count <= fp_budget_count:
            recall_at_budget = float(recall_score(y_true, y_p, zero_division=0))
            break

    return {
        "pr_auc":              round(float(average_precision_score(y_true, y_prob)), 6),
        "roc_auc":             round(float(roc_auc_score(y_true, y_prob)), 6),
        "accuracy":            round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":           round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":              round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score":            round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "recall_at_fp_budget": round(recall_at_budget, 6),
        "fp_budget_rate":      fp_budget,
        "threshold":           round(threshold, 4),
        "n_samples":           int(len(y_true)),
        "n_fraud":             int(y_true.sum()),
        "fraud_rate":          round(float(y_true.mean()), 6),
    }


def subgroup_metrics(df: pd.DataFrame, y_true, y_prob, threshold: float, col: str, label: str) -> list:
    """Return per-group PR-AUC and recall for a given grouping column."""
    rows = []
    for group_val, group_idx in df.groupby(col).groups.items():
        yt = y_true[group_idx.values]
        yp = y_prob[group_idx.values]
        if yt.sum() < 2:
            continue   # need at least 2 fraud cases for PR-AUC
        y_pred = (yp >= threshold).astype(int)
        rows.append({
            "group_dimension": label,
            "group_value":     str(group_val),
            "n_samples":       int(len(yt)),
            "n_fraud":         int(yt.sum()),
            "fraud_rate":      round(float(yt.mean()), 6),
            "pr_auc":          round(float(average_precision_score(yt, yp)), 6),
            "recall":          round(float(recall_score(yt, y_pred, zero_division=0)), 6),
            "precision":       round(float(precision_score(yt, y_pred, zero_division=0)), 6),
            "f1_score":        round(float(f1_score(yt, y_pred, zero_division=0)), 6),
        })
    return rows


def main() -> None:
    logger.info("=== CreditOps Model Evaluation ===")
    logger.info("NOTICE: Source data is simulated (Sparkov). No real cardholders involved.")

    # ── Load artifacts ──────────────────────────────────────────────────── #
    with (ROOT_DIR / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (ROOT_DIR / "preprocessor.pkl").open("rb") as f:
        preprocessor = pickle.load(f)

    threshold_path = ROOT_DIR / ".decision_threshold"
    threshold = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5
    logger.info("Decision threshold: %.4f", threshold)

    # ── Load test split (raw, then apply features) ──────────────────────── #
    test_raw_path = ROOT_DIR / "data" / "raw" / "test.csv"
    if not test_raw_path.exists():
        raise FileNotFoundError("data/raw/test.csv not found. Run dvc repro Data_Collection first.")

    test_raw = pd.read_csv(test_raw_path, low_memory=False)
    y_test   = test_raw[TARGET].values

    X_test_features = build_features(test_raw)
    X_test_enc      = preprocessor.transform(X_test_features)
    test_prob       = model.predict_proba(X_test_enc)[:, 1]

    # ── Overall metrics ─────────────────────────────────────────────────── #
    metrics = compute_metrics(y_test, test_prob, threshold=threshold)
    logger.info(
        "Test | PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | recall=%.4f | recall@FPR=%.4f",
        metrics["pr_auc"], metrics["roc_auc"], metrics["f1_score"],
        metrics["recall"], metrics["recall_at_fp_budget"],
    )

    # ── Subgroup evaluation ──────────────────────────────────────────────── #
    test_raw_reset = test_raw.reset_index(drop=True)
    y_arr  = np.array(y_test)
    sg_rows = []

    # By category
    sg_rows += subgroup_metrics(test_raw_reset, y_arr, test_prob, threshold, "category", "category")

    # By amount bucket
    test_raw_reset["amt_bucket"] = pd.cut(
        test_raw_reset["amt"],
        bins=[0, 10, 50, 100, 500, float("inf")],
        labels=["$0-10", "$10-50", "$50-100", "$100-500", "$500+"],
    )
    sg_rows += subgroup_metrics(test_raw_reset, y_arr, test_prob, threshold, "amt_bucket", "amt_bucket")

    # By age band (derive from raw)
    ts    = pd.to_datetime(test_raw_reset["trans_date_trans_time"])
    dob   = pd.to_datetime(test_raw_reset["dob"], errors="coerce")
    ages  = ((ts - dob).dt.days / 365.25).clip(lower=0, upper=120).fillna(45)
    test_raw_reset["age_band"] = pd.cut(
        ages, bins=[0, 25, 35, 50, 65, 200],
        labels=["<25", "25-35", "35-50", "50-65", "65+"],
    )
    sg_rows += subgroup_metrics(test_raw_reset, y_arr, test_prob, threshold, "age_band", "age_band")

    # By hour of day
    test_raw_reset["hour"] = ts.dt.hour
    test_raw_reset["hour_band"] = pd.cut(
        test_raw_reset["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night(0-5)", "Morning(6-11)", "Afternoon(12-17)", "Evening(18-23)"],
    )
    sg_rows += subgroup_metrics(test_raw_reset, y_arr, test_prob, threshold, "hour_band", "hour_of_day")

    # By haversine distance bucket
    import math
    def hvs(row):
        try:
            R = 6371.0
            phi1, phi2 = math.radians(row.lat), math.radians(row.merch_lat)
            dphi = math.radians(row.merch_lat - row.lat)
            dlam = math.radians(row.merch_long - row["long"])
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        except Exception:
            return 0.0

    if all(c in test_raw_reset.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        test_raw_reset["dist_km"] = test_raw_reset.apply(hvs, axis=1)
        test_raw_reset["dist_bucket"] = pd.cut(
            test_raw_reset["dist_km"],
            bins=[0, 10, 50, 100, float("inf")],
            labels=["<10km", "10-50km", "50-100km", ">100km"],
        )
        sg_rows += subgroup_metrics(test_raw_reset, y_arr, test_prob, threshold, "dist_bucket", "distance_bucket")

    subgroup_df = pd.DataFrame(sg_rows)
    subgroup_path = ROOT_DIR / "subgroup_metrics.csv"
    subgroup_df.to_csv(subgroup_path, index=False)
    logger.info("subgroup_metrics.csv written: %d rows", len(subgroup_df))

    # Weakest subgroups
    if not subgroup_df.empty:
        weakest = subgroup_df.nsmallest(5, "pr_auc")[
            ["group_dimension", "group_value", "n_samples", "pr_auc", "recall"]
        ]
        logger.info("Weakest subgroups (by PR-AUC):\n%s", weakest.to_string(index=False))
        metrics["weakest_subgroups"] = weakest.to_dict("records")

    metrics["data_provenance"] = "synthetic/simulated (Sparkov)"
    metrics["model_type"] = type(model).__name__

    out = ROOT_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("metrics.json written to %s", out)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()