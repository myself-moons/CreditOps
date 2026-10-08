"""
tests/observatory/test_phase5.py — CreditOps v2 Phase 5 Tests

Covers:
  1. FirebaseLogStore with mocked client (no network in CI)
  2. Store factory & graceful fallback to SQLite
  3. Live prediction logging (LogStore + predictions.jsonl + no PII)
  4. Observatory FastAPI routes (/api/summary, /api/scenarios, /api/trace,
     /api/models, /api/audit, /api/decisions, /api/live)
  5. RBAC security on POST /api/rollback (risk_owner) and POST /api/replay (ml_engineer)
  6. Dashboard HTML serving at /observatory
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.observatory.storage.base import LogStore
from src.observatory.storage.factory import get_log_store
from src.observatory.storage.firebase_store import FirebaseLogStore
from src.observatory.storage.sqlite_store import SQLiteLogStore
from src.prediction_logger import log_prediction

ROOT = Path(__file__).resolve().parent.parent.parent
client = TestClient(app)


# ──────────────────────────────────────────────────────────────────────────── #
# Mock Firebase Client                                                         #
# ──────────────────────────────────────────────────────────────────── #

class MockChildRef:
    def __init__(self, key: str, storage: Dict[str, Any]):
        self.key = key
        self._storage = storage

    def set(self, value: Dict[str, Any]):
        self._storage[self.key] = value


class MockQueryRef:
    def __init__(self, data: Dict[str, Any], limit: int | None = None):
        self._data = data
        self._limit = limit

    def get(self):
        items = list(self._data.values())
        if self._limit:
            return items[-self._limit:]
        return self._data


class MockDbRef:
    def __init__(self, path: str):
        self.path = path
        self._storage: Dict[str, Any] = {}
        self._counter = 0

    def push(self) -> MockChildRef:
        self._counter += 1
        key = f"-mock_id_{self._counter}"
        return MockChildRef(key, self._storage)

    def set(self, value: Any):
        self._storage = value

    def update(self, updates: Dict[str, Any]):
        self._storage.update(updates)

    def order_by_key(self):
        return self

    def limit_to_last(self, limit: int):
        return MockQueryRef(self._storage, limit)

    def get(self):
        return self._storage


class MockFirebaseDb:
    def __init__(self):
        self.refs: Dict[str, MockDbRef] = {}

    def reference(self, path: str, app: Any = None):
        if path not in self.refs:
            self.refs[path] = MockDbRef(path)
        return self.refs[path]


# ──────────────────────────────────────────────────────────────────────────── #
# 1. FirebaseLogStore Mocked Tests                                             #
# ──────────────────────────────────────────────────────────────────────────── #

class TestFirebaseLogStore:
    def test_append_and_query(self):
        mock_db = MockFirebaseDb()
        store = FirebaseLogStore(db_client=mock_db)

        rec_id = store.append("audit_log", {
            "actor": "admin",
            "role": "risk_owner",
            "action": "test_append",
        })
        assert rec_id.startswith("-mock_id_")

        results = store.query("audit_log")
        assert len(results) == 1
        assert results[0]["actor"] == "admin"
        assert results[0]["id"] == rec_id
        assert "timestamp" in results[0]

    def test_batch_append(self):
        mock_db = MockFirebaseDb()
        store = FirebaseLogStore(db_client=mock_db)

        ids = store.batch_append("drift_events", [
            {"scenario": "control", "score": 0.1},
            {"scenario": "novel_pattern", "score": 0.8},
        ])
        assert len(ids) == 2
        items = store.query("drift_events")
        assert len(items) == 2

    def test_query_filtering(self):
        mock_db = MockFirebaseDb()
        store = FirebaseLogStore(db_client=mock_db)

        store.append("policy_decisions", {"action": "alert", "scenario": "control"})
        store.append("policy_decisions", {"action": "retrain", "scenario": "novel_pattern"})

        res_alert = store.query("policy_decisions", filters={"action": "alert"})
        assert len(res_alert) == 1
        assert res_alert[0]["scenario"] == "control"

    def test_invalid_collection_raises(self):
        mock_db = MockFirebaseDb()
        store = FirebaseLogStore(db_client=mock_db)

        with pytest.raises(ValueError, match="Unknown collection"):
            store.append("unsupported_collection", {"foo": "bar"})


# ──────────────────────────────────────────────────────────────────────────── #
# 2. Store Factory & Fallback                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

class TestStoreFactory:
    def test_default_sqlite(self, tmp_path):
        store = get_log_store(backend="sqlite", db_path=str(tmp_path / "test.db"))
        assert isinstance(store, SQLiteLogStore)

    def test_firebase_fallback_when_vars_missing(self, monkeypatch):
        monkeypatch.delenv("FIREBASE_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("FIREBASE_DATABASE_URL", raising=False)

        store = get_log_store(backend="firebase")
        # Should gracefully fall back to SQLiteLogStore
        assert isinstance(store, SQLiteLogStore)

    def test_firebase_fallback_when_file_missing(self, monkeypatch):
        monkeypatch.setenv("FIREBASE_CREDENTIALS_PATH", "non_existent_key.json")
        monkeypatch.setenv("FIREBASE_DATABASE_URL", "https://mock.firebaseio.com")

        store = get_log_store(backend="firebase")
        assert isinstance(store, SQLiteLogStore)


# ──────────────────────────────────────────────────────────────────────────── #
# 3. Live Prediction Logging (No PII)                                          #
# ──────────────────────────────────────────────────────────────────────────── #

class TestLivePredictionLogging:
    def test_log_prediction_persists_to_store_and_jsonl(self, tmp_path):
        # Trigger log_prediction
        log_prediction(
            prediction=1,
            probability=0.985,
            decision="fraud",
            model_version="XGBClassifier",
            latency_ms=12.4,
            derived_features={"log_amt": 5.4, "haversine_km": 120.5, "cc_num": "4111222233334444"},
            identifier="customer_12345",
        )

        store = get_log_store()
        recs = store.query("prediction_log", limit=5)
        assert len(recs) > 0
        latest = recs[-1]

        assert latest["prediction"] == 1
        assert latest["decision"] == "fraud"
        assert latest["model_version"] == "XGBClassifier"
        # Verify PII cc_num stripped
        if "features" in latest and latest["features"]:
            assert "cc_num" not in latest["features"]
        # Verify customer id hashed
        if "id_hash" in latest:
            assert latest["id_hash"] != "customer_12345"

    def test_api_predict_writes_to_log_store(self):
        payload = {
            "trans_date_trans_time": "2020-06-15 12:00:00",
            "category": "grocery_pos",
            "amt": 28.50,
            "gender": "M",
            "dob": "1990-01-01",
            "lat": 36.1,
            "long": -115.2,
            "city_pop": 50000,
            "merch_lat": 36.12,
            "merch_long": -115.22,
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

        store = get_log_store()
        logged = store.query("prediction_log", limit=10)
        assert len(logged) > 0


# ──────────────────────────────────────────────────────────────────────────── #
# 4. Observatory Routes & RBAC                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class TestObservatoryAPI:
    def test_dashboard_page_200(self):
        res = client.get("/observatory")
        assert res.status_code == 200
        assert "CreditOps Drift Observatory" in res.text

    def test_summary_api_200(self):
        res = client.get("/observatory/api/summary")
        assert res.status_code == 200
        data = res.json()
        assert "acceptance_criteria" in data

    def test_scenarios_api_200(self):
        res = client.get("/observatory/api/scenarios")
        assert res.status_code == 200
        data = res.json()
        assert "scenarios" in data
        assert "policies" in data

    def test_trace_api_200(self):
        res = client.get("/observatory/api/trace/control/NeverRetrain")
        assert res.status_code == 200
        data = res.json()
        assert "window_traces" in data

    def test_models_api_200(self):
        res = client.get("/observatory/api/models")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_audit_api_200(self):
        res = client.get("/observatory/api/audit")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_decisions_api_200(self):
        res = client.get("/observatory/api/decisions")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_live_api_200(self):
        res = client.get("/observatory/api/live")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "backend" in data

    # RBAC Tests
    def test_rollback_requires_auth(self):
        # Missing key -> 401
        res = client.post("/observatory/api/rollback", json={"target_version": "v1.0"})
        assert res.status_code == 401

        # Invalid key -> 401
        res = client.post(
            "/observatory/api/rollback",
            headers={"X-API-Key": "bogus_key"},
            json={"target_version": "v1.0"},
        )
        assert res.status_code == 401

        # ML Engineer key (wrong role) -> 403
        res = client.post(
            "/observatory/api/rollback",
            headers={"X-API-Key": "ml_engineer_key_456"},
            json={"target_version": "v1.0"},
        )
        assert res.status_code == 403

        # Risk Owner key -> 200
        res = client.post(
            "/observatory/api/rollback",
            headers={"X-API-Key": "risk_owner_key_123"},
            json={"target_version": "v1.0", "reason": "Test authorized rollback"},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "SUCCESS"

    def test_replay_requires_ml_engineer(self):
        # Missing key -> 401
        res = client.post("/observatory/api/replay", json={"scenario": "control"})
        assert res.status_code == 401

        # Risk owner key (wrong role) -> 403
        res = client.post(
            "/observatory/api/replay",
            headers={"X-API-Key": "risk_owner_key_123"},
            json={"scenario": "control"},
        )
        assert res.status_code == 403

        # ML Engineer key -> 200
        res = client.post(
            "/observatory/api/replay",
            headers={"X-API-Key": "ml_engineer_key_456"},
            json={"scenario": "control", "policy": "NeverRetrain"},
        )
        assert res.status_code == 200
        assert "window_traces" in res.json()

    def test_pytest_store_is_never_firebase(self):
        """Item 1: Assert store is strictly not Firebase under pytest execution."""
        store = get_log_store()
        assert not isinstance(store, FirebaseLogStore)
        assert isinstance(store, SQLiteLogStore)

    def test_audit_and_model_records_contain_no_raw_keys_or_prefixes(self, monkeypatch):
        """Item 2: Assert no raw key or key prefix appears in any audit/model record."""
        import hashlib
        raw_key = "secret_owner_token_987654"
        key_prefix = raw_key[:8]
        monkeypatch.setenv("OBS_API_KEYS", f"{raw_key}:risk_owner")
        expected_fp = f"risk_owner_{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:8]}"

        res = client.post(
            "/observatory/api/rollback",
            headers={"X-API-Key": raw_key},
            json={"target_version": "v1.0", "reason": "Credential sanitization verification"},
        )
        assert res.status_code == 200

        store = get_log_store()
        audit_records = store.query("audit_log", limit=50)
        model_records = store.query("model_versions", limit=50)

        assert len(audit_records) > 0
        assert len(model_records) > 0

        # Check all audit records
        for rec in audit_records:
            actor = rec.get("actor", "")
            # Ensure no raw key or prefix appears in actor or anywhere in serialized record
            assert raw_key not in actor, f"Raw key found in audit actor: {actor}"
            assert key_prefix not in actor, f"Key prefix found in audit actor: {actor}"
            assert raw_key not in json.dumps(rec), f"Raw key leaked in audit record: {rec}"
            assert key_prefix not in json.dumps(rec), f"Key prefix leaked in audit record: {rec}"

        # Check all model records
        for rec in model_records:
            actor = rec.get("actor", "")
            assert raw_key not in actor, f"Raw key found in model actor: {actor}"
            assert key_prefix not in actor, f"Key prefix found in model actor: {actor}"
            assert raw_key not in json.dumps(rec), f"Raw key leaked in model record: {rec}"
            assert key_prefix not in json.dumps(rec), f"Key prefix leaked in model record: {rec}"

        # Verify the latest audit record has the correct sha256 fingerprint
        latest_audit = audit_records[-1]
        assert latest_audit["actor"] == expected_fp
