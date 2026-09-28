"""
monitor.py — CreditOps

TWO-LAYER MONITORING ARCHITECTURE
----------------------------------
Layer A — Operational monitoring (no ground truth required):
  - prediction count, fraud detection rate, latency, model version
  - probability distribution stats
  - feature distribution vs training profile (check_distribution_drift)

Layer B — Model performance monitoring (requires ground-truth labels):
  - PR-AUC, recall, precision, F1
  - Called when labelled outcomes are available (offline workflow)

IMPORTANT: Without ground-truth labels, PR-AUC cannot be computed.
Distribution shift is an early warning only.

⚠ SIMULATED DATA: All monitoring compares against a Sparkov-simulation baseline.
  Any drift detected is simulation-internal variation, not real-world concept drift.
"""

import json
import statistics
from pathlib import Path
from typing import Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "predictions.jsonl"


def _load_logs() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_monitoring_cfg() -> dict:
    params_path = ROOT_DIR / "params.yaml"
    if not params_path.exists():
        return {}
    with params_path.open() as f:
        return yaml.safe_load(f).get("monitoring", {})


def get_operational_stats(recent_n: int = 100) -> dict:
    """
    Layer A: Operational statistics from prediction log.
    No ground-truth labels required.

    Returns stats for the most recent recent_n predictions (or all if recent_n < 0).
    """
    records = _load_logs()
    if not records:
        return {
            "status":            "no_predictions",
            "message":           "No predictions have been logged yet.",
            "total_predictions": 0,
            "data_provenance":   "synthetic/simulated (Sparkov)",
        }

    window = records[-recent_n:] if recent_n > 0 else records
    total  = len(records)

    predictions  = [r["prediction"] for r in window]
    probs        = [r["probability"] for r in window]
    latencies    = [r["latency_ms"]  for r in window if "latency_ms" in r]
    model_vers   = {}
    for r in window:
        mv = r.get("model_version", "unknown")
        model_vers[mv] = model_vers.get(mv, 0) + 1

    fraud_rate   = sum(predictions) / len(predictions) if predictions else 0.0
    sorted_lat   = sorted(latencies)
    p95_latency  = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else None

    return {
        "status":              "ok",
        "data_provenance":     "synthetic/simulated (Sparkov)",
        "total_predictions":   total,
        "window_size":         len(window),
        "fraud_detection_rate": round(fraud_rate, 4),
        "fraud_count":         sum(predictions),
        "legitimate_count":    len(predictions) - sum(predictions),
        "probability_stats": {
            "mean": round(statistics.mean(probs), 4) if probs else None,
            "min":  round(min(probs), 4)             if probs else None,
            "max":  round(max(probs), 4)             if probs else None,
            "std":  round(statistics.stdev(probs), 4) if len(probs) > 1 else 0.0,
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "p95":  round(p95_latency, 2)                if p95_latency else None,
        },
        "model_versions": model_vers,
        "layer_a_note": (
            "Operational stats only. PR-AUC, recall, and F1 require ground-truth "
            "fraud labels and cannot be computed from prediction logs alone."
        ),
    }


def evaluate_with_ground_truth(labeled_outcomes: list[dict]) -> Optional[dict]:
    """
    Layer B: Performance evaluation using ground-truth labels.

    Args:
        labeled_outcomes: List of dicts:
            {"timestamp": "<ISO matching a log record>", "actual_label": 0|1}

    Returns:
        Performance metrics dict, or None if insufficient data.
    """
    try:
        from sklearn.metrics import (
            average_precision_score, f1_score,
            precision_score, recall_score, roc_auc_score,
        )
    except ImportError:
        return {"error": "sklearn not available"}

    logs    = _load_logs()
    log_map = {r["timestamp"]: r for r in logs}

    y_true, y_pred, y_prob = [], [], []
    for outcome in labeled_outcomes:
        ts = outcome.get("timestamp")
        if ts in log_map:
            y_true.append(outcome["actual_label"])
            y_pred.append(log_map[ts]["prediction"])
            y_prob.append(log_map[ts]["probability"])

    if len(y_true) < 10:
        return {
            "status":  "insufficient_data",
            "matched": len(y_true),
            "message": "Need at least 10 matched ground-truth labels for evaluation.",
        }

    return {
        "status":    "evaluated",
        "n_samples": len(y_true),
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 6),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score":  round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "data_provenance": "synthetic/simulated (Sparkov)",
    }
