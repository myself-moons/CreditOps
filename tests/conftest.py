"""
tests/conftest.py — Global Test Configuration & Environment Isolation

Ensures tests never connect to or mutate live Firebase Realtime Database.
Executes before any test module or application module is imported.
"""

import os
import sys
from pathlib import Path

# Force SQLite and clear/block Firebase env vars before src.main or log store is imported
os.environ["STORE_BACKEND"] = "sqlite"
os.environ["FIREBASE_CREDENTIALS_PATH"] = ""
os.environ["FIREBASE_DATABASE_URL"] = ""

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from src.observatory.storage import get_log_store
from src.observatory.storage.sqlite_store import SQLiteLogStore
from src.observatory.storage.firebase_store import FirebaseLogStore


@pytest.fixture(autouse=True)
def ensure_sqlite_in_tests(monkeypatch):
    """Enforce SQLiteLogStore in all tests."""
    monkeypatch.setenv("STORE_BACKEND", "sqlite")
    monkeypatch.setenv("FIREBASE_CREDENTIALS_PATH", "")
    monkeypatch.setenv("FIREBASE_DATABASE_URL", "")
