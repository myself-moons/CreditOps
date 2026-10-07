"""
src/observatory/storage/sqlite_store.py — CreditOps v2 Observatory

SQLite implementation of LogStore.
Stores events as JSON payloads indexed by id, collection, and ISO timestamp.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.observatory.storage.base import LogStore, VALID_COLLECTIONS

logger = logging.getLogger(__name__)


class _JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy and datetime types safely."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, Path)):
            return str(obj)
        return super().default(obj)


class SQLiteLogStore(LogStore):
    """
    SQLite-backed event and log store.

    Table schema:
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            collection TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            payload TEXT NOT NULL
        )
    """

    def __init__(self, db_path: Union[str, Path] = "runs/observatory.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_coll_ts ON records(collection, timestamp)"
            )

    def append(self, collection: str, record: Dict[str, Any]) -> str:
        if collection not in VALID_COLLECTIONS:
            raise ValueError(
                f"Unknown collection '{collection}'. Valid collections: {sorted(VALID_COLLECTIONS)}"
            )

        payload_dict = dict(record)
        rec_id = str(payload_dict.get("id") or uuid.uuid4())
        ts = str(
            payload_dict.get("timestamp")
            or datetime.now(timezone.utc).isoformat()
        )
        payload_dict["id"] = rec_id
        payload_dict["timestamp"] = ts
        payload_dict["collection"] = collection

        payload_json = json.dumps(payload_dict, cls=_JSONEncoder)

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO records (id, collection, timestamp, payload)
                VALUES (?, ?, ?, ?)
                """,
                (rec_id, collection, ts, payload_json),
            )

        return rec_id

    def query(
        self,
        collection: str,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if collection not in VALID_COLLECTIONS:
            raise ValueError(
                f"Unknown collection '{collection}'. Valid collections: {sorted(VALID_COLLECTIONS)}"
            )

        query_sql = "SELECT payload FROM records WHERE collection = ? ORDER BY timestamp ASC"
        params: list[Any] = [collection]

        cursor = self._conn.cursor()
        cursor.execute(query_sql, params)
        rows = cursor.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            record = json.loads(row["payload"])
            if filters:
                match = True
                for k, v in filters.items():
                    if record.get(k) != v and record.get("details", {}).get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            results.append(record)
            if limit is not None and len(results) >= limit:
                break

        return results

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def __enter__(self) -> SQLiteLogStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
