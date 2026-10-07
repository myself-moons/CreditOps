"""
src/observatory/policies/base.py — CreditOps v2 Retraining Policies Base

Defines standard interface and state containers for retraining policies:
  decide(state) -> PolicyDecision(action in {"no_action", "alert", "retrain"}, reason, signals)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.observatory.monitoring.detectors import DetectionResult


VALID_ACTIONS = frozenset({"no_action", "alert", "retrain"})


@dataclass
class PolicyDecision:
    """Action outcome returned by a policy at window t."""
    action: str  # "no_action" | "alert" | "retrain"
    reason: str
    signals: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(f"Invalid policy action '{self.action}'. Valid: {sorted(VALID_ACTIONS)}")


@dataclass
class PolicyState:
    """
    Context provided to policy at window t to make a retraining decision:
      - window_index: current stream window t
      - current_detector_results: {detector_name: DetectionResult} at window t
      - detector_history: list of detector results from window 0..t
      - performance_result: latest PerformanceMonitor DetectionResult
      - performance_history: list of PerformanceMonitor DetectionResult from 0..t
      - windows_since_last_retrain: int count of windows since most recent retrain
      - recent_costs: recent per-window decision costs
      - reference_cost_per_window: baseline cost per window from reference period
    """
    window_index: int
    current_detector_results: Dict[str, DetectionResult] = field(default_factory=dict)
    detector_history: List[Dict[str, DetectionResult]] = field(default_factory=list)
    performance_result: Optional[DetectionResult] = None
    performance_history: List[DetectionResult] = field(default_factory=list)
    windows_since_last_retrain: int = 999
    recent_costs: List[float] = field(default_factory=list)
    reference_cost_per_window: float = 0.0


class BasePolicy(ABC):
    """Abstract base class for all retraining policies."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def decide(self, state: PolicyState) -> PolicyDecision:
        """Evaluate state and return a PolicyDecision."""
        raise NotImplementedError
