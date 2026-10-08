"""
src/prediction_logger.py — CreditOps

Appends one JSON record per prediction to predictions.jsonl and writes to LogStore
(collection 'prediction_log') through the store factory so records survive server restarts.

⚠ PII note: raw transaction PII (card number, name, street) is NEVER logged.
Any identifiers are hashed with SHA-256. Only model-relevant features, score, decision,
and timestamp are stored.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "predictions.jsonl"

logger = logging.getLogger(__name__)


def log_prediction(
    *,
    prediction: int,
    probability: float,
    decision: str,
    model_version: str,
    latency_ms: float,
    derived_features: Optional[Dict[str, Any]] = None,
    identifier: Optional[str] = None,
) -> None:
    """
    Append a single prediction record to predictions.jsonl and LogStore (prediction_log).
    """
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    record: Dict[str, Any] = {
        "timestamp": timestamp_iso,
        "prediction": int(prediction),
        "probability": round(float(probability), 6),
        "score": round(float(probability), 6),
        "decision": decision,
        "model_version": model_version,
        "latency_ms": round(float(latency_ms), 3),
    }

    if identifier:
        record["id_hash"] = hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()[:16]

    if derived_features is not None:
        # Strip any personal / non-model fields
        clean_features = {}
        for k, v in derived_features.items():
            if k in {"first", "last", "cc_num", "street", "dob", "trans_num", "card_number", "name", "address"}:
                continue
            clean_features[k] = (round(v, 6) if isinstance(v, float) else int(v) if isinstance(v, (int, bool)) else v)
        record["derived_features"] = clean_features

    # 1. Keep predictions.jsonl behaviour working
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Failed appending to predictions.jsonl: %s", e)

    # 2. Write to LogStore through store factory
    try:
        from src.observatory.storage import get_log_store

        store = get_log_store()
        store_entry = {
            "timestamp": timestamp_iso,
            "score": round(float(probability), 6),
            "decision": decision,
            "prediction": int(prediction),
            "model_version": model_version,
            "latency_ms": round(float(latency_ms), 3),
            "features": record.get("derived_features", {}),
        }
        if "id_hash" in record:
            store_entry["id_hash"] = record["id_hash"]

        store.append("prediction_log", store_entry)
    except Exception as e:
        logger.warning("Failed appending to LogStore prediction_log: %s", e)
