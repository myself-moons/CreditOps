"""
src/observatory/run_stream.py — CreditOps v2 Observatory, Phase 1

CLI script executed by DVC stage 'Observatory_Drift' and experiments.
Generates stream windows according to observatory.yaml and writes:
  - runs/<run_id>/manifest.json
  - runs/<run_id>/stream_summary.json
  - runs/latest/manifest.json
  - runs/latest/stream_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.observatory.config import load_config
from src.observatory.simulator.stream import WindowedStream
from src.observatory.simulator.manifest import write_scenario_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_stream")


def load_raw_stream_data(cfg) -> pd.DataFrame:
    """Load real Sparkov transaction data via DatasetAdapter sliced to stream_start..stream_end."""
    from src.dataset_adapter import DatasetAdapter
    adapter = DatasetAdapter(train_path=cfg.raw_train, test_path=cfg.raw_test)
    df = adapter.load_slice(cfg.stream_start, cfg.stream_end)
    if df.empty:
        raise ValueError(
            f"DatasetAdapter loaded 0 rows for stream period {cfg.stream_start}..{cfg.stream_end}. "
            f"Ensure data/fraudTest.csv exists."
        )
    return df


def main(config_path: Optional[Path] = None, run_id: Optional[str] = None) -> int:
    cfg = load_config(config_path)
    logger.info("Loaded config: scenario=%s, shape=%s, seed=%d",
                cfg.drift_injection.scenario, cfg.drift_injection.shape, cfg.seed)

    raw_df = load_raw_stream_data(cfg)
    stream = WindowedStream(raw_df=raw_df, config=cfg)

    window_summaries = []
    total_fraud = 0
    total_txns = 0

    for window in stream:
        total_txns += window.n_rows
        total_fraud += window.n_fraud
        window_summaries.append({
            "window_index": window.window_index,
            "start_date": window.start_date,
            "end_date": window.end_date,
            "n_rows": window.n_rows,
            "n_fraud": window.n_fraud,
            "fraud_rate": window.fraud_rate,
            "is_drifted": window.is_drifted,
            "drift_strength": window.drift_strength,
        })

    n_windows = len(window_summaries)
    logger.info("Stream processing finished. Yielded %d windows with %d total transactions (%d fraud).",
                n_windows, total_txns, total_fraud)

    # Write scenario manifest
    manifest_path = write_scenario_manifest(
        config=cfg,
        n_windows=n_windows,
        source_df=raw_df,
        run_id=run_id,
    )

    resolved_run_id = manifest_path.parent.name
    summary_data = {
        "run_id": resolved_run_id,
        "n_windows": n_windows,
        "total_rows": total_txns,
        "total_fraud": total_fraud,
        "mean_fraud_rate": (total_fraud / total_txns) if total_txns > 0 else 0.0,
        "windows": window_summaries,
    }

    # Write stream_summary.json
    run_dir = manifest_path.parent
    summary_path = run_dir / "stream_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    latest_dir = ROOT_DIR / "runs" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    with (latest_dir / "stream_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Stream summary saved to %s and runs/latest/stream_summary.json", summary_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drift observatory stream generation")
    parser.add_argument("--config", type=Path, default=None, help="Path to observatory.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run identifier")
    args = parser.parse_args()
    sys.exit(main(args.config, args.run_id))
