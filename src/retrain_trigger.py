"""
retrain_trigger.py — CreditOps

Evaluates whether model retraining should be recommended.

Primary trigger: PR-AUC < params.yaml monitoring.pr_auc_threshold
Secondary trigger: recall < params.yaml monitoring.recall_threshold

Distribution drift check compares recent fraud prediction rate against
the training distribution recorded in params.yaml.

⚠ SIMULATED DATA: Any detected drift is simulation-internal variation.
  Real-world drift experiments must be explicitly injected.
"""

from pathlib import Path
from typing import Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_monitoring_cfg() -> dict:
    params_path = ROOT_DIR / "params.yaml"
    if not params_path.exists():
        return {}
    with params_path.open() as f:
        return yaml.safe_load(f).get("monitoring", {})


def evaluate_with_labels(labeled_outcomes: list[dict]) -> dict:
    """
    Layer B trigger: Compute PR-AUC from ground-truth labels and decide
    whether retraining is needed.

    Returns dict with retrain_recommended, reason, pr_auc, recall, threshold.
    """
    try:
        from src.monitor import evaluate_with_ground_truth
    except ImportError:
        from monitor import evaluate_with_ground_truth

    cfg       = _load_monitoring_cfg()
    pr_thresh = float(cfg.get("pr_auc_threshold",  0.10))
    re_thresh = float(cfg.get("recall_threshold",  0.60))
    min_preds = int(cfg.get("min_predictions",     50))

    result = evaluate_with_ground_truth(labeled_outcomes)
    if result is None or result.get("status") != "evaluated":
        return {
            "retrain_recommended": False,
            "reason": result.get("message", "Insufficient ground-truth data.") if result else "No data.",
            "pr_auc_threshold":  pr_thresh,
            "recall_threshold":  re_thresh,
            "data_provenance":   "synthetic/simulated (Sparkov)",
        }

    pr_auc  = result["pr_auc"]
    recall  = result["recall"]
    retrain = (pr_auc < pr_thresh) or (recall < re_thresh)

    reasons = []
    if pr_auc < pr_thresh:
        reasons.append(f"PR-AUC {pr_auc:.4f} < threshold {pr_thresh:.4f}")
    if recall < re_thresh:
        reasons.append(f"recall {recall:.4f} < threshold {re_thresh:.4f}")

    return {
        "retrain_recommended": retrain,
        "reason":              "; ".join(reasons) if reasons else "Metrics above thresholds — no action needed.",
        "pr_auc":              pr_auc,
        "recall":              recall,
        "pr_auc_threshold":    pr_thresh,
        "recall_threshold":    re_thresh,
        "n_samples":           result.get("n_samples"),
        "performance":         result,
        "data_provenance":     "synthetic/simulated (Sparkov)",
    }


def check_distribution_drift(recent_n: int = 100) -> dict:
    """
    Layer A weak signal: Compare recent fraud prediction rate against
    the training distribution baseline stored in params.yaml.

    This is an early-warning signal only.
    It does NOT measure model performance without ground-truth labels.

    ⚠ NOTE: The training distribution is from Sparkov-simulated data.
             Any shift detected is simulation-internal variation.
    """
    try:
        from src.monitor import get_operational_stats
    except ImportError:
        from monitor import get_operational_stats

    cfg              = _load_monitoring_cfg()
    training_fr      = float(cfg.get("training_fraud_rate", 0.00579))
    drift_tolerance  = float(cfg.get("drift_tolerance",     0.003))

    stats = get_operational_stats(recent_n=recent_n)
    if stats.get("status") != "ok":
        return {
            "drift_detected":  False,
            "reason":          "No prediction data available.",
            "data_provenance": "synthetic/simulated (Sparkov)",
        }

    recent_fr = stats["fraud_detection_rate"]
    deviation = abs(recent_fr - training_fr)
    drift     = deviation > drift_tolerance

    return {
        "drift_detected":            drift,
        "training_fraud_rate":       training_fr,
        "recent_fraud_detection_rate": recent_fr,
        "absolute_deviation":        round(deviation, 6),
        "tolerance":                 drift_tolerance,
        "total_predictions":         stats["total_predictions"],
        "data_provenance":           "synthetic/simulated (Sparkov)",
        "warning": (
            "Prediction distribution has shifted vs training baseline. "
            "This is a weak signal — ground-truth evaluation needed to confirm "
            "whether model performance has degraded. "
            "Note: baseline is from simulated data; any shift is simulation-internal."
        ) if drift else None,
        "note": (
            "Distribution drift is a weak signal only. "
            "True performance monitoring requires ground-truth labels."
        ),
    }
