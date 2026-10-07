"""
src/observatory/registry.py — CreditOps v2 Model Registry & State Machine

Manages candidate, promoted, rejected, and rolled_back model versions.
Enforces strict lifecycle state transitions:
    candidate -> promoted | rejected
    promoted -> rolled_back
Invalid transitions raise InvalidStateTransitionError.

Writes model_versions and audit_log records through the LogStore.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.observatory.storage.base import LogStore

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal lifecycle transition is attempted."""
    pass


# Strict allowed transitions mapping: from_state -> set(to_state)
ALLOWED_TRANSITIONS = {
    ModelState.CANDIDATE: {ModelState.PROMOTED, ModelState.REJECTED},
    ModelState.PROMOTED: {ModelState.ROLLED_BACK},
    ModelState.REJECTED: set(),
    ModelState.ROLLED_BACK: set(),
}


@dataclass
class ModelRecord:
    """Record of a model registered in the registry."""
    version: str
    state: ModelState
    model: Any
    threshold: float
    data_hash: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_window: int = 0
    history: List[str] = field(default_factory=list)

    def transition_to(self, new_state: ModelState) -> None:
        """Validate and apply state transition."""
        allowed = ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"Cannot transition model '{self.version}' from '{self.state.value}' to '{new_state.value}'. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )
        self.state = new_state
        self.history.append(new_state.value)


class ModelRegistry:
    """
    Central model registry and state machine with LogStore auditing.
    """

    def __init__(self, log_store: Optional[LogStore] = None) -> None:
        self.log_store = log_store
        self.models: Dict[str, ModelRecord] = {}
        self.active_champion: Optional[ModelRecord] = None
        self.shadow_champion: Optional[ModelRecord] = None

    def register_initial_champion(
        self,
        model: Any,
        version: str = "v1.0",
        threshold: float = 0.97,
        data_hash: str = "base_v1_data",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ModelRecord:
        """Register the baseline V1 champion directly as PROMOTED."""
        record = ModelRecord(
            version=version,
            state=ModelState.PROMOTED,
            model=model,
            threshold=float(threshold),
            data_hash=data_hash,
            metrics=metrics or {},
            created_window=0,
            history=[ModelState.PROMOTED.value],
        )
        self.models[version] = record
        self.active_champion = record
        self._log_audit(
            actor="system",
            action="init_champion",
            version=version,
            data_hash=data_hash,
            metrics=metrics or {},
            reason="Initialized baseline V1 champion",
        )
        return record

    def register_candidate(
        self,
        model: Any,
        version: str,
        threshold: float,
        data_hash: str,
        metrics: Optional[Dict[str, Any]] = None,
        actor: str = "governance_engine",
        reason: str = "Retrained challenger on labelled stream windows",
        created_window: int = 0,
    ) -> ModelRecord:
        """Register a newly trained challenger in CANDIDATE state."""
        if version in self.models:
            raise ValueError(f"Model version '{version}' already exists in registry.")

        record = ModelRecord(
            version=version,
            state=ModelState.CANDIDATE,
            model=model,
            threshold=float(threshold),
            data_hash=data_hash,
            metrics=metrics or {},
            created_window=created_window,
            history=[ModelState.CANDIDATE.value],
        )
        self.models[version] = record

        self._log_audit(
            actor=actor,
            action="register_candidate",
            version=version,
            data_hash=data_hash,
            metrics=metrics or {},
            reason=reason,
        )
        return record

    def promote(
        self,
        version: str,
        actor: str = "promotion_gate",
        reason: str = "Passed promotion gate criteria",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ModelRecord:
        """Promote a candidate to active champion and move current champion to shadow."""
        record = self.models.get(version)
        if record is None:
            raise KeyError(f"Model version '{version}' not found.")

        # Validates candidate -> promoted
        record.transition_to(ModelState.PROMOTED)
        if metrics:
            record.metrics.update(metrics)

        # Move current active champion to shadow
        self.shadow_champion = self.active_champion
        self.active_champion = record

        self._log_audit(
            actor=actor,
            action="promote",
            version=version,
            data_hash=record.data_hash,
            metrics=record.metrics,
            reason=reason,
        )
        return record

    def reject(
        self,
        version: str,
        actor: str = "promotion_gate",
        reason: str = "Failed promotion gate criteria",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ModelRecord:
        """Reject a candidate."""
        record = self.models.get(version)
        if record is None:
            raise KeyError(f"Model version '{version}' not found.")

        # Validates candidate -> rejected
        record.transition_to(ModelState.REJECTED)
        if metrics:
            record.metrics.update(metrics)

        self._log_audit(
            actor=actor,
            action="reject",
            version=version,
            data_hash=record.data_hash,
            metrics=record.metrics,
            reason=reason,
        )
        return record

    def rollback(
        self,
        version: str,
        actor: str = "rollback_monitor",
        reason: str = "Exceeded cost degradation margin",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ModelRecord:
        """Roll back a promoted model and restore previous champion from shadow."""
        record = self.models.get(version)
        if record is None:
            raise KeyError(f"Model version '{version}' not found.")

        # Validates promoted -> rolled_back
        record.transition_to(ModelState.ROLLED_BACK)
        if metrics:
            record.metrics.update(metrics)

        if self.shadow_champion is not None:
            self.active_champion = self.shadow_champion
            self.shadow_champion = None

        self._log_audit(
            actor=actor,
            action="rollback",
            version=version,
            data_hash=record.data_hash,
            metrics=record.metrics,
            reason=reason,
        )
        return record

    def get_model(self, version: str) -> Optional[ModelRecord]:
        return self.models.get(version)

    def _log_audit(
        self,
        actor: str,
        action: str,
        version: str,
        data_hash: str,
        metrics: Dict[str, Any],
        reason: str,
    ) -> None:
        """Log state changes to model_versions and audit_log collections in LogStore."""
        if self.log_store is None:
            return

        payload = {
            "actor": actor,
            "action": action,
            "model_version": version,
            "data_hash": data_hash,
            "metrics": metrics,
            "reason": reason,
        }

        try:
            self.log_store.append("model_versions", payload)
            self.log_store.append("audit_log", payload)
        except Exception as e:
            logger.warning("Failed to write to LogStore: %s", e)
