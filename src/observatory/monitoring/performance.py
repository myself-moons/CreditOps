"""
src/observatory/monitoring/performance.py — CreditOps v2 Observatory

PerformanceMonitor with configurable label delay (default 2 windows).
At window t, it evaluates the champion model using labels from windows <= t - delay
over the last K_perf windows (default 5).

Reports:
  - PR-AUC
  - Recall at V1 decision threshold
  - Recall broken down by merchant category
  - Robust handling of insufficient data (returns NaN without crashing)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from src.observatory.monitoring.detectors import (
    CATEGORIES,
    DetectionResult,
)
from src.observatory.simulator.stream import StreamWindow
from src.observatory.storage.base import LogStore

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def _load_v1_decision_threshold() -> float:
    thresh_file = ROOT_DIR / ".decision_threshold"
    if thresh_file.exists():
        try:
            return float(thresh_file.read_text(encoding="utf-8").strip())
        except Exception as e:
            logger.warning("Could not read .decision_threshold (%s) — using 0.97", e)
    return 0.97


class PerformanceMonitor:
    """
    Monitors model performance over a rolling window of delayed labels.
    """

    def __init__(
        self,
        champion_model: Any,
        preprocessor: Any,
        label_delay_windows: int = 2,
        k_perf: int = 5,
        min_fraud_rows: int = 5,
        decision_threshold: Optional[float] = None,
        threshold_recall: float = 0.50,
        reference_pr_auc: Optional[float] = None,
        threshold_pr_auc_drop: float = 0.180,
        log_store: Optional[LogStore] = None,
    ) -> None:
        self.champion_model = champion_model
        self.preprocessor = preprocessor
        self.label_delay_windows = max(0, int(label_delay_windows))
        self.k_perf = max(1, int(k_perf))
        self.min_fraud_rows = max(1, int(min_fraud_rows))
        self.decision_threshold = (
            float(decision_threshold)
            if decision_threshold is not None
            else _load_v1_decision_threshold()
        )
        self.threshold_recall = float(threshold_recall)
        self.reference_pr_auc = float(reference_pr_auc) if reference_pr_auc is not None else None
        self.threshold_pr_auc_drop = float(threshold_pr_auc_drop)
        self.log_store = log_store
        self.name = "performance_monitor"

        # Buffer of past StreamWindow objects indexed by window_index
        self._window_history: Dict[int, StreamWindow] = {}

    def _log_failure(self, window_index: int, error_msg: str) -> DetectionResult:
        logger.warning("[%s] Robustness fallback at window %d: %s", self.name, window_index, error_msg)
        res = DetectionResult(
            score=float("nan"),
            alarm=False,
            window_index=window_index,
            detector=self.name,
            details={"error": error_msg, "status": "error"},
        )
        if self.log_store is not None:
            try:
                self.log_store.append("drift_events", {
                    "detector": self.name,
                    "window_index": window_index,
                    "alarm": False,
                    "score": None,
                    "error": error_msg,
                })
            except Exception as e:
                logger.error("Failed to append to log_store: %s", e)
        return res

    def update(self, window: StreamWindow) -> DetectionResult:
        if window is None or window.features_df is None or len(window.features_df) == 0:
            return self._log_failure(getattr(window, "window_index", -1), "Empty window or missing features")

        t = window.window_index
        self._window_history[t] = window

        # Eligible labeled windows: w <= t - label_delay_windows
        max_eligible_w = t - self.label_delay_windows
        if max_eligible_w < 0:
            return DetectionResult(
                score=float("nan"),
                alarm=False,
                window_index=t,
                detector=self.name,
                details={
                    "status": "insufficient_data",
                    "reason": f"Waiting for label delay (t={t} < delay={self.label_delay_windows})",
                    "pr_auc": float("nan"),
                    "recall": float("nan"),
                },
            )

        # Rolling K_perf windows ending at max_eligible_w
        min_eligible_w = max(0, max_eligible_w - self.k_perf + 1)
        eval_window_indices = [
            w_idx for w_idx in range(min_eligible_w, max_eligible_w + 1)
            if w_idx in self._window_history
        ]

        if not eval_window_indices:
            return DetectionResult(
                score=float("nan"),
                alarm=False,
                window_index=t,
                detector=self.name,
                details={
                    "status": "insufficient_data",
                    "reason": "No evaluation windows found in history",
                    "pr_auc": float("nan"),
                    "recall": float("nan"),
                },
            )

        try:
            eval_windows = [self._window_history[w_idx] for w_idx in eval_window_indices]
            eval_feats = pd.concat([w.features_df for w in eval_windows], ignore_index=True)
            eval_labels = pd.concat([w.labels for w in eval_windows], ignore_index=True)

            n_fraud = int(eval_labels.sum())
            if n_fraud < self.min_fraud_rows:
                return DetectionResult(
                    score=float("nan"),
                    alarm=False,
                    window_index=t,
                    detector=self.name,
                    details={
                        "status": "insufficient_data",
                        "reason": f"Fraud count ({n_fraud}) < min_fraud_rows ({self.min_fraud_rows})",
                        "num_fraud": n_fraud,
                        "num_rows": len(eval_feats),
                        "pr_auc": float("nan"),
                        "recall": float("nan"),
                        "evaluated_windows": eval_window_indices,
                    },
                )

            # Predict probabilities
            X_enc = self.preprocessor.transform(eval_feats)
            probs = self.champion_model.predict_proba(X_enc)[:, 1]

            # Calculate PR-AUC
            precision, recall_curve, _ = precision_recall_curve(eval_labels, probs)
            pr_auc = float(auc(recall_curve, precision))

            # Calculate Recall at decision threshold
            preds = (probs >= self.decision_threshold).astype(int)
            tp = int(((preds == 1) & (eval_labels == 1)).sum())
            overall_recall = float(tp / n_fraud) if n_fraud > 0 else 0.0

            # Calculate recall by merchant category
            recall_by_cat: Dict[str, Optional[float]] = {}
            for cat in CATEGORIES:
                col = f"cat_{cat}"
                if col in eval_feats.columns:
                    cat_mask = (eval_feats[col] == 1)
                    cat_fraud = int(((eval_labels == 1) & cat_mask).sum())
                    if cat_fraud > 0:
                        cat_tp = int(((preds == 1) & (eval_labels == 1) & cat_mask).sum())
                        recall_by_cat[cat] = round(float(cat_tp / cat_fraud), 4)
                    else:
                        recall_by_cat[cat] = None
                else:
                    recall_by_cat[cat] = None

            # Calculate alarms: recall degradation OR PR-AUC drop relative to reference
            recall_alarm = bool(overall_recall < self.threshold_recall)
            if self.reference_pr_auc is not None:
                pr_auc_drop = float(self.reference_pr_auc - pr_auc)
                pr_auc_alarm = bool(pr_auc_drop >= self.threshold_pr_auc_drop)
            else:
                pr_auc_drop = 0.0
                pr_auc_alarm = False

            alarm = bool(recall_alarm or pr_auc_alarm)

            details = {
                "status": "ok",
                "pr_auc": round(pr_auc, 4),
                "recall": round(overall_recall, 4),
                "reference_pr_auc": round(self.reference_pr_auc, 4) if self.reference_pr_auc is not None else None,
                "pr_auc_drop": round(pr_auc_drop, 4),
                "threshold_recall": self.threshold_recall,
                "threshold_pr_auc_drop": self.threshold_pr_auc_drop,
                "recall_alarm": recall_alarm,
                "pr_auc_alarm": pr_auc_alarm,
                "decision_threshold": self.decision_threshold,
                "num_rows": len(eval_feats),
                "num_fraud": n_fraud,
                "evaluated_windows": eval_window_indices,
                "recall_by_category": recall_by_cat,
            }

            if self.log_store is not None and alarm:
                self.log_store.append("drift_events", {
                    "detector": self.name,
                    "window_index": t,
                    "alarm": alarm,
                    "score": overall_recall,
                    "details": details,
                })

            return DetectionResult(
                score=overall_recall,
                alarm=alarm,
                window_index=t,
                detector=self.name,
                details=details,
            )

        except Exception as exc:
            return self._log_failure(t, f"Performance calculation error: {str(exc)}")
