"""
src/observatory/policies/threshold_policy.py — Threshold Trigger Policy

Retrains on alarm of the DataDriftDetector (V1-style PSI/drift trigger).
"""

from __future__ import annotations

from src.observatory.policies.base import BasePolicy, PolicyDecision, PolicyState


class ThresholdPolicy(BasePolicy):
    """
    V1-style baseline policy that retrains upon observing an alarm
    from the DataDriftDetector, subject to cooldown.
    """

    def __init__(self, cooldown_windows: int = 5) -> None:
        super().__init__(name="ThresholdPolicy")
        self.cooldown_windows = max(0, int(cooldown_windows))

    def decide(self, state: PolicyState) -> PolicyDecision:
        # Check cooldown
        if state.windows_since_last_retrain < self.cooldown_windows:
            return PolicyDecision(
                action="no_action",
                reason=f"Cooldown active ({state.windows_since_last_retrain} < {self.cooldown_windows})",
                signals={"windows_since_last_retrain": state.windows_since_last_retrain},
            )

        dd_result = state.current_detector_results.get("DataDrift") or state.current_detector_results.get("data_drift")
        if dd_result is not None and dd_result.alarm:
            return PolicyDecision(
                action="retrain",
                reason=f"DataDriftDetector alarmed (score={dd_result.score:.4f} >= threshold)",
                signals={
                    "data_drift_alarm": True,
                    "data_drift_score": dd_result.score,
                    "windows_since_last_retrain": state.windows_since_last_retrain,
                },
            )

        return PolicyDecision(
            action="no_action",
            reason="DataDriftDetector did not alarm",
            signals={
                "data_drift_alarm": False,
                "data_drift_score": dd_result.score if dd_result else None,
            },
        )
