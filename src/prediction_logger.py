"""
prediction_logger.py — CreditOps

Appends one JSON record per prediction to predictions.jsonl.

⚠ PII note: raw transaction PII (card number, name, street) is NOT logged.
  Only derived feature values and prediction results are stored.

⚠ SIMULATED DATA: This logger supports a model trained on Sparkov simulation data.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "predictions.jsonl"


def log_prediction(
    *,
    prediction: int,
    probability: float,
    decision: str,
    model_version: str,
    latency_ms: float,
    derived_features: Optional[dict] = None,
) -> None:
    """
    Append a single prediction record to predictions.jsonl.

    Args:
        prediction:       0 (legitimate) or 1 (fraud)
        probability:      Model fraud score (0.0–1.0)
        decision:         Human-readable decision ("fraud" or "legitimate")
        model_version:    Model class name, e.g. 'XGBClassifier'
        latency_ms:       End-to-end latency in milliseconds
        derived_features: Feature values computed by features.py (no PII)
    """
    record: dict = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "prediction":    int(prediction),
        "probability":   round(float(probability), 6),
        "decision":      decision,
        "model_version": model_version,
        "latency_ms":    round(float(latency_ms), 3),
    }
    if derived_features is not None:
        record["derived_features"] = {
            k: (round(v, 6) if isinstance(v, float) else int(v) if isinstance(v, (int, bool)) else v)
            for k, v in derived_features.items()
        }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
