"""
src/observatory/policies/fixed_schedule.py — Fixed Schedule Retraining Policy

Retrains on a fixed periodic schedule every K windows regardless of drift signals.
"""

from __future__ import annotations

from src.observatory.policies.base import BasePolicy, PolicyDecision, PolicyState


class FixedSchedule(BasePolicy):
    """Retrains periodically every k_windows."""

    def __init__(self, k_windows: int = 15) -> None:
        super().__init__(name="FixedSchedule")
        self.k_windows = max(1, int(k_windows))

    def decide(self, state: PolicyState) -> PolicyDecision:
        if state.windows_since_last_retrain >= self.k_windows:
            return PolicyDecision(
                action="retrain",
                reason=f"Fixed schedule interval reached ({state.windows_since_last_retrain} >= {self.k_windows})",
                signals={
                    "windows_since_last_retrain": state.windows_since_last_retrain,
                    "k_windows": self.k_windows,
                },
            )

        return PolicyDecision(
            action="no_action",
            reason=f"Waiting for fixed schedule ({state.windows_since_last_retrain} < {self.k_windows})",
            signals={
                "windows_since_last_retrain": state.windows_since_last_retrain,
                "k_windows": self.k_windows,
            },
        )
