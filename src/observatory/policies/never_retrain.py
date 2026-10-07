"""
src/observatory/policies/never_retrain.py — NeverRetrain Policy Baseline

Never triggers retraining or alert actions (serves as the static champion baseline).
"""

from __future__ import annotations

from src.observatory.policies.base import BasePolicy, PolicyDecision, PolicyState


class NeverRetrain(BasePolicy):
    """Static baseline policy that never retrains or raises alerts."""

    def __init__(self) -> None:
        super().__init__(name="NeverRetrain")

    def decide(self, state: PolicyState) -> PolicyDecision:
        return PolicyDecision(
            action="no_action",
            reason="NeverRetrain policy active (static baseline)",
            signals={"windows_since_last_retrain": state.windows_since_last_retrain},
        )
