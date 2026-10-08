"""
src/observatory/storage/factory.py — CreditOps v2 LogStore Factory

Provides a unified factory to instantiate the configured LogStore backend:
STORE_BACKEND=firebase|sqlite (default: sqlite)
Falls back gracefully to SQLiteLogStore if Firebase configuration is missing or network fails.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.observatory.storage.base import LogStore
from src.observatory.storage.sqlite_store import SQLiteLogStore

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(ROOT / ".env")


def get_log_store(backend: Optional[str] = None, **kwargs) -> LogStore:
    """
    Instantiate LogStore according to STORE_BACKEND or fallback.

    Args:
        backend: 'firebase' or 'sqlite'. If None, reads from STORE_BACKEND env var.
        kwargs: Optional arguments passed to the specific backend.

    Returns:
        LogStore instance (FirebaseLogStore or SQLiteLogStore).
    """
    if backend is None:
        backend = os.getenv("STORE_BACKEND", "sqlite").strip().lower()

    if backend == "firebase":
        try:
            from src.observatory.storage.firebase_store import FirebaseLogStore

            cred_json_env = os.getenv("FIREBASE_CREDENTIALS_JSON")
            cred_path_env = os.getenv("FIREBASE_CREDENTIALS_PATH")
            db_url_env = os.getenv("FIREBASE_DATABASE_URL")

            if (not cred_path_env and not cred_json_env) or not db_url_env:
                logger.warning(
                    "STORE_BACKEND is 'firebase' but neither FIREBASE_CREDENTIALS_PATH nor "
                    "FIREBASE_CREDENTIALS_JSON (or FIREBASE_DATABASE_URL) is set. Falling back to SQLite."
                )
                return _fallback_sqlite(**kwargs)

            if cred_json_env:
                return FirebaseLogStore(
                    credentials_path=None,
                    database_url=db_url_env,
                    **kwargs,
                )

            # Resolve relative path from project root
            cred_path = Path(cred_path_env)
            if not cred_path.is_absolute():
                cred_path = (ROOT / cred_path).resolve()

            if not cred_path.exists():
                logger.warning(
                    "Firebase credentials file not found at '%s'. Falling back to SQLite.",
                    cred_path,
                )
                return _fallback_sqlite(**kwargs)

            return FirebaseLogStore(
                credentials_path=cred_path,
                database_url=db_url_env,
                **kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize FirebaseLogStore (%s). Falling back to SQLite.",
                exc,
            )
            return _fallback_sqlite(**kwargs)

    return _fallback_sqlite(**kwargs)


def _fallback_sqlite(**kwargs) -> SQLiteLogStore:
    default_db = ROOT / "runs" / "observatory.db"
    db_path = kwargs.get("db_path", str(default_db))
    return SQLiteLogStore(db_path=db_path)
