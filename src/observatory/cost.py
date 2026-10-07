"""
src/observatory/cost.py — CreditOps v2 Observatory Cost Model

Calculates operational and model decision costs according to the business specification:
  - fn_cost: cost per missed fraud (False Negative, default $500)
  - fp_cost: cost per false alarm / review (False Positive, default $5)
  - review_cost: cost per manual review triggered by a policy alert (default $10)
  - retrain_cost: cost per retraining run (default $200)

Per-window decision cost = FN * fn_cost + FP * fp_cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd


@dataclass
class WindowCostResult:
    """Breakdown of financial costs for a single stream window."""
    window_index: int
    n_samples: int
    n_fraud: int
    tp: int
    fp: int
    tn: int
    fn: int
    fn_cost: float
    fp_cost: float
    decision_cost: float     # fn * fn_cost + fp * fp_cost
    retrain_cost: float      # charged if policy chose retrain
    alert_cost: float        # charged if policy chose alert
    total_cost: float        # decision_cost + retrain_cost + alert_cost


class CostModel:
    """
    Evaluates window costs based on champion predictions and ground truth labels.
    """

    def __init__(
        self,
        fn_cost: float = 500.0,
        fp_cost: float = 5.0,
        retrain_cost: float = 200.0,
        review_cost: float = 10.0,
    ) -> None:
        self.fn_cost = float(fn_cost)
        self.fp_cost = float(fp_cost)
        self.retrain_cost = float(retrain_cost)
        self.review_cost = float(review_cost)

    def evaluate_window(
        self,
        window_index: int,
        y_true: Union[pd.Series, np.ndarray, Sequence[int]],
        y_prob: Union[pd.Series, np.ndarray, Sequence[float]],
        decision_threshold: float = 0.97,
        action: str = "no_action",
    ) -> WindowCostResult:
        """
        Evaluate classification metrics and monetary costs for one window.
        """
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_prob, dtype=float)

        if len(y_t) == 0:
            return WindowCostResult(
                window_index=window_index,
                n_samples=0,
                n_fraud=0,
                tp=0, fp=0, tn=0, fn=0,
                fn_cost=0.0, fp_cost=0.0,
                decision_cost=0.0,
                retrain_cost=self.retrain_cost if action == "retrain" else 0.0,
                alert_cost=self.review_cost if action == "alert" else 0.0,
                total_cost=(self.retrain_cost if action == "retrain" else 0.0) +
                           (self.review_cost if action == "alert" else 0.0),
            )

        y_pred = (y_p >= decision_threshold).astype(int)

        tp = int(((y_pred == 1) & (y_t == 1)).sum())
        fp = int(((y_pred == 1) & (y_t == 0)).sum())
        tn = int(((y_pred == 0) & (y_t == 0)).sum())
        fn = int(((y_pred == 0) & (y_t == 1)).sum())

        fn_cost_total = fn * self.fn_cost
        fp_cost_total = fp * self.fp_cost
        decision_cost = fn_cost_total + fp_cost_total

        retrain_cost_charged = self.retrain_cost if action == "retrain" else 0.0
        alert_cost_charged = self.review_cost if action == "alert" else 0.0
        total_cost = decision_cost + retrain_cost_charged + alert_cost_charged

        return WindowCostResult(
            window_index=window_index,
            n_samples=len(y_t),
            n_fraud=int(y_t.sum()),
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            fn_cost=fn_cost_total,
            fp_cost=fp_cost_total,
            decision_cost=decision_cost,
            retrain_cost=retrain_cost_charged,
            alert_cost=alert_cost_charged,
            total_cost=total_cost,
        )
