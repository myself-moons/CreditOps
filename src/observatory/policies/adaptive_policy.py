"""
src/observatory/policies/adaptive_policy.py — Adaptive Retraining Policy

Evaluates multiple multi-modal drift signals:
  - Label-free: Data drift, fraud-score drift, unseen categories, isolation outliers
  - Delayed labels: Performance monitor alarms (PR-AUC / Recall drop)

Retraining rule:
  (performance alarm persists N_perf windows)
  OR
  (label-free alarms persist N_free windows AND expected_benefit > retrain_cost)

Subject to cooldown of C windows after any retrain.
Expected benefit = (estimated excess per-window cost vs reference) * horizon_windows (H).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.observatory.monitoring.detectors import DetectionResult
from src.observatory.policies.base import BasePolicy, PolicyDecision, PolicyState

logger = logging.getLogger(__name__)


class AdaptivePolicy(BasePolicy):
    """
    Cost-aware adaptive retraining policy with persistence filtering,
    label-delay awareness, and cooldown gating.
    """

    LABEL_FREE_NAMES = frozenset({
        "DataDrift", "data_drift",
        "UnseenCategory", "unseen_category",
        "IsolationOutlier", "isolation_outlier",
        "FraudScoreDrift", "fraud_score_drift",
        "Novelty", "novelty",
    })

    def __init__(
        self,
        n_perf_persistence: int = 2,
        n_free_persistence: int = 2,
        cooldown_windows: int = 5,
        horizon_windows: int = 10,
        retrain_cost: float = 200.0,
    ) -> None:
        super().__init__(name="AdaptivePolicy")
        self.n_perf = max(1, int(n_perf_persistence))
        self.n_free = max(1, int(n_free_persistence))
        self.cooldown = max(0, int(cooldown_windows))
        self.horizon = max(1, int(horizon_windows))
        self.retrain_cost = float(retrain_cost)

    def _has_label_free_alarm(self, results: Dict[str, DetectionResult]) -> bool:
        """Check if any label-free detector alarmed in a given window result dictionary."""
        return any(
            res.alarm for name, res in results.items()
            if name in self.LABEL_FREE_NAMES
        )

    def decide(self, state: PolicyState) -> PolicyDecision:
        signals: Dict[str, Any] = {
            "windows_since_last_retrain": state.windows_since_last_retrain,
            "cooldown": self.cooldown,
            "n_perf": self.n_perf,
            "n_free": self.n_free,
            "horizon": self.horizon,
            "retrain_cost": self.retrain_cost,
        }

        # 1. Cooldown check
        if state.windows_since_last_retrain < self.cooldown:
            return PolicyDecision(
                action="no_action",
                reason=f"Cooldown active ({state.windows_since_last_retrain} < {self.cooldown} windows)",
                signals=signals,
            )

        # 2. Check Performance Alarm Persistence (Delayed Ground Truth)
        perf_history = list(state.performance_history)
        if state.performance_result is not None and (not perf_history or perf_history[-1] != state.performance_result):
            perf_history.append(state.performance_result)

        if len(perf_history) >= self.n_perf:
            recent_perf = perf_history[-self.n_perf:]
            if all(r.alarm for r in recent_perf):
                signals["performance_persistence"] = True
                signals["recent_perf_scores"] = [r.score for r in recent_perf]
                return PolicyDecision(
                    action="retrain",
                    reason=f"Performance degradation persisted for {self.n_perf} consecutive windows",
                    signals=signals,
                )

        # 3. Check Label-Free Alarm Persistence & Expected Benefit Gate
        det_history = list(state.detector_history)
        if state.current_detector_results and (not det_history or det_history[-1] != state.current_detector_results):
            det_history.append(state.current_detector_results)

        label_free_persisted = False
        if len(det_history) >= self.n_free:
            recent_det_history = det_history[-self.n_free:]
            label_free_persisted = all(self._has_label_free_alarm(d) for d in recent_det_history)

        signals["label_free_persisted"] = label_free_persisted

        if label_free_persisted:
            # Estimate excess cost per window vs reference
            if state.recent_costs:
                current_cost_est = float(np.mean(state.recent_costs[-5:]))
            else:
                current_cost_est = state.reference_cost_per_window

            excess_cost_per_window = max(0.0, current_cost_est - state.reference_cost_per_window)
            
            # If an unseen category or major structural drift is detected, calculate expected benefit
            unseen_res = state.current_detector_results.get("UnseenCategory") or state.current_detector_results.get("unseen_category")
            if unseen_res and unseen_res.alarm:
                # Unseen category represents structural gap; estimate excess cost of at least 1 missed fraud per window
                excess_cost_per_window = max(excess_cost_per_window, 100.0)

            expected_benefit = excess_cost_per_window * self.horizon
            signals["current_cost_est"] = current_cost_est
            signals["reference_cost_per_window"] = state.reference_cost_per_window
            signals["excess_cost_per_window"] = excess_cost_per_window
            signals["expected_benefit"] = expected_benefit

            if expected_benefit > self.retrain_cost:
                return PolicyDecision(
                    action="retrain",
                    reason=(
                        f"Label-free alarms persisted for {self.n_free} windows with expected benefit "
                        f"${expected_benefit:.2f} > retrain cost ${self.retrain_cost:.2f}"
                    ),
                    signals=signals,
                )

        # 4. If any detector alarmed right now, raise alert (e.g. for monitoring triage)
        any_current_alarm = (
            self._has_label_free_alarm(state.current_detector_results) or
            (state.performance_result is not None and state.performance_result.alarm)
        )

        if any_current_alarm:
            return PolicyDecision(
                action="alert",
                reason="Drift detected; monitoring persistence or awaiting cost benefit clearance",
                signals=signals,
            )

        # 5. Default: no action
        return PolicyDecision(
            action="no_action",
            reason="All signals nominal; no retrain or alert required",
            signals=signals,
        )
