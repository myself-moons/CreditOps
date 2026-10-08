"""
src/observatory/storage/firebase_store.py — CreditOps v2 Firebase Realtime Database LogStore

Implements LogStore interface using firebase-admin SDK.
Stores records under /observatory/<collection>/<push-id>.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.observatory.storage.base import LogStore, VALID_COLLECTIONS

logger = logging.getLogger(__name__)


class FirebaseLogStore(LogStore):
    """
    LogStore implementation backed by Firebase Realtime Database.
    Collections stored under /observatory/<collection>/<push-id>.
    """

    def __init__(
        self,
        credentials_path: Optional[str | Path] = None,
        database_url: Optional[str] = None,
        db_client: Any = None,
        app_name: str = "creditops_observatory",
    ) -> None:
        self.app_name = app_name
        self._db = db_client

        if self._db is None:
            self._init_firebase(credentials_path, database_url)

    def _init_firebase(
        self,
        credentials_path: Optional[str | Path],
        database_url: Optional[str],
    ) -> None:
        import firebase_admin
        from firebase_admin import credentials, db

        cred_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if not credentials_path and not cred_json_str:
            credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
        if not database_url:
            database_url = os.getenv("FIREBASE_DATABASE_URL")

        if (not credentials_path and not cred_json_str) or not database_url:
            raise ValueError(
                "Missing Firebase configuration: (FIREBASE_CREDENTIALS_PATH or "
                "FIREBASE_CREDENTIALS_JSON) and FIREBASE_DATABASE_URL must be provided or set in environment."
            )

        if not cred_json_str:
            # Resolve relative paths from project root if needed
            cred_path = Path(credentials_path)
            if not cred_path.is_absolute():
                # Try from cwd or project root
                root_cand = Path(__file__).resolve().parent.parent.parent.parent
                if (root_cand / cred_path).exists():
                    cred_path = root_cand / cred_path
                else:
                    cred_path = Path.cwd() / cred_path

            if not cred_path.exists():
                raise FileNotFoundError(f"Firebase credentials file not found: {cred_path}")

        try:
            # Check if app already initialized
            app = None
            try:
                app = firebase_admin.get_app(self.app_name)
            except ValueError:
                pass

            if app is None:
                if cred_json_str:
                    try:
                        cert_data = json.loads(cred_json_str)
                    except Exception as json_err:
                        raise ValueError(f"Invalid JSON in FIREBASE_CREDENTIALS_JSON: {json_err}")
                    cred = credentials.Certificate(cert_data)
                else:
                    cred = credentials.Certificate(str(cred_path))
                app = firebase_admin.initialize_app(
                    cred,
                    {"databaseURL": database_url},
                    name=self.app_name,
                )

            self._app = app
            self._db = db
            logger.info("Firebase Realtime Database initialized (app=%s, url=%s)", self.app_name, database_url)
        except Exception as e:
            logger.error("Failed to initialize Firebase app: %s", e)
            raise ConnectionError(f"Firebase initialization failed: {e}") from e

    def _get_ref(self, collection: str):
        if collection not in VALID_COLLECTIONS:
            raise ValueError(f"Unknown collection '{collection}'. Must be one of {sorted(VALID_COLLECTIONS)}")
        return self._db.reference(f"observatory/{collection}", app=getattr(self, "_app", None))

    def append(self, collection: str, record: Dict[str, Any]) -> str:
        rec = dict(record)
        if "timestamp" not in rec:
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()

        ref = self._get_ref(collection)
        new_ref = ref.push()
        record_id = new_ref.key
        if "id" not in rec:
            rec["id"] = record_id

        new_ref.set(rec)
        return str(record_id)

    def batch_append(self, collection: str, records: List[Dict[str, Any]]) -> List[str]:
        if not records:
            return []
        ref = self._get_ref(collection)
        updates = {}
        ids = []
        for r in records:
            rec = dict(r)
            if "timestamp" not in rec:
                rec["timestamp"] = datetime.now(timezone.utc).isoformat()
            new_child = ref.push()
            rec_id = new_child.key
            if "id" not in rec:
                rec["id"] = rec_id
            updates[rec_id] = rec
            ids.append(rec_id)

        ref.update(updates)
        return ids

    def query(
        self,
        collection: str,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ref = self._get_ref(collection)
        try:
            if limit and hasattr(ref, "order_by_key") and hasattr(ref, "limit_to_last"):
                query_ref = ref.order_by_key().limit_to_last(limit)
                raw_data = query_ref.get()
            elif hasattr(ref, "get"):
                raw_data = ref.get()
            else:
                raw_data = None
        except Exception as e:
            logger.warning("Error querying Firebase collection '%s': %s", collection, e)
            return []

        if not raw_data:
            return []

        results: List[Dict[str, Any]] = []
        if isinstance(raw_data, dict):
            for k, v in raw_data.items():
                if isinstance(v, dict):
                    item = dict(v)
                    if "id" not in item:
                        item["id"] = k
                    results.append(item)
        elif isinstance(raw_data, list):
            for v in raw_data:
                if isinstance(v, dict):
                    results.append(dict(v))

        # Apply in-memory equality filters if provided
        if filters:
            filtered = []
            for item in results:
                match = True
                for fk, fv in filters.items():
                    if item.get(fk) != fv:
                        match = False
                        break
                if match:
                    filtered.append(item)
            results = filtered

        # If limit was provided and not handled by DB or post-filter
        if limit and len(results) > limit:
            results = results[-limit:]

        return results
