"""
src/observatory/storage/base.py — CreditOps v2 Observatory

Abstract interface for append-only and queryable event storage.
Collections supported across Phase 2-4:
  - drift_events       (Phase 2)
  - policy_decisions   (Phase 3)
  - model_versions     (Phase 3)
  - audit_log          (Phase 4)
  - prediction_log     (Phase 4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

VALID_COLLECTIONS = frozenset(
    {
        "drift_events",
        "policy_decisions",
        "model_versions",
        "audit_log",
        "prediction_log",
    }
)


class LogStore(ABC):
    """
    Abstract interface for Observatory log storage.
    Backends implement append and query across valid collections.
    """

    @abstractmethod
    def append(self, collection: str, record: Dict[str, Any]) -> str:
        """
        Append a record to the specified collection.

        Args:
            collection: One of VALID_COLLECTIONS.
            record: Dict containing event payload. If 'id' or 'timestamp'
                    are omitted, backend will populate them.

        Returns:
            The record ID (str).
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        collection: str,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query records from a collection.

        Args:
            collection: One of VALID_COLLECTIONS.
            limit: Maximum number of records to return.
            filters: Key-value equality filters applied to record attributes.

        Returns:
            List of matching record dicts.
        """
        raise NotImplementedError
