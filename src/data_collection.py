"""
data_collection.py — CreditOps

DVC Stage: Data_Collection
--------------------------
Reads Credit_Data/fraudTrain.csv and Credit_Data/fraudTest.csv,
concatenates, parses timestamps, drops exact duplicates, validates schema,
sorts by time, then applies a TEMPORAL split (no random splitting):
  - First 70%  → data/raw/train.csv
  - Next  15%  → data/raw/val.csv
  - Last  15%  → data/raw/test.csv

Also writes split_info.json with date ranges and fraud rates per split.

SIMULATED DATA NOTICE
---------------------
The source data is a Sparkov simulation. No real cardholders are involved.
Fraud rate in the full dataset is approximately 0.5%.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset_adapter import DatasetAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    params_path = ROOT_DIR / "params.yaml"
    with params_path.open() as f:
        params = yaml.safe_load(f)

    data_cfg  = params.get("data", {})
    train_frac = float(data_cfg.get("train_frac", 0.70))
    val_frac   = float(data_cfg.get("val_frac",   0.15))

    adapter = DatasetAdapter(params)

    logger.info("=== CreditOps Data Collection ===")
    logger.info("NOTICE: Source data is simulated (Sparkov). No real cardholders involved.")

    df = adapter.load()

    logger.info("Splitting temporally: %.0f%% train / %.0f%% val / %.0f%% test",
                train_frac * 100, val_frac * 100, (1 - train_frac - val_frac) * 100)

    train, val, test = adapter.split_by_time(df, train_frac=train_frac, val_frac=val_frac)

    time_col  = adapter.time_column
    label_col = adapter.label_column

    # Build split info JSON (real numbers from real data)
    split_info = {
        "data_provenance": DatasetAdapter.DATA_PROVENANCE,
        "note": "All numbers below are measured from the real (simulated) data.",
        "total_rows": int(len(df)),
        "splits": {
            "train": {
                "rows": int(len(train)),
                "date_min": str(train[time_col].min()),
                "date_max": str(train[time_col].max()),
                "fraud_rate": round(float(train[label_col].mean()), 6),
                "fraud_count": int(train[label_col].sum()),
            },
            "val": {
                "rows": int(len(val)),
                "date_min": str(val[time_col].min()),
                "date_max": str(val[time_col].max()),
                "fraud_rate": round(float(val[label_col].mean()), 6),
                "fraud_count": int(val[label_col].sum()),
            },
            "test": {
                "rows": int(len(test)),
                "date_min": str(test[time_col].min()),
                "date_max": str(test[time_col].max()),
                "fraud_rate": round(float(test[label_col].mean()), 6),
                "fraud_count": int(test[label_col].sum()),
            },
        },
    }

    logger.info("Train : %d rows | fraud rate %.4f%% | %s → %s",
                split_info["splits"]["train"]["rows"],
                split_info["splits"]["train"]["fraud_rate"] * 100,
                split_info["splits"]["train"]["date_min"],
                split_info["splits"]["train"]["date_max"])
    logger.info("Val   : %d rows | fraud rate %.4f%% | %s → %s",
                split_info["splits"]["val"]["rows"],
                split_info["splits"]["val"]["fraud_rate"] * 100,
                split_info["splits"]["val"]["date_min"],
                split_info["splits"]["val"]["date_max"])
    logger.info("Test  : %d rows | fraud rate %.4f%% | %s → %s",
                split_info["splits"]["test"]["rows"],
                split_info["splits"]["test"]["fraud_rate"] * 100,
                split_info["splits"]["test"]["date_min"],
                split_info["splits"]["test"]["date_max"])

    out_dir = ROOT_DIR / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv",   index=False)
    test.to_csv(out_dir / "test.csv",  index=False)

    split_info_path = ROOT_DIR / "split_info.json"
    split_info_path.write_text(json.dumps(split_info, indent=2, default=str))

    logger.info("split_info.json written to %s", split_info_path)
    logger.info("Data collection complete.")


if __name__ == "__main__":
    main()