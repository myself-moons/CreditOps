"""
src/observatory/simulator/manifest.py — CreditOps v2 Observatory, Phase 1

Scenario manifest generation for offline simulation runs.

For every run, writes runs/<run_id>/manifest.json with:
  - scenario
  - shape
  - seed
  - onset_window
  - ramp_windows
  - magnitude
  - window_size_days
  - n_windows
  - data hash of source slice
  - git commit

Ensures exact reproducibility: same seed + config yields identical manifest and windows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def get_git_commit() -> str:
    """Retrieve the current git commit hash, or 'unknown' if not in git."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "unknown"


def compute_slice_hash(df: pd.DataFrame) -> str:
    """
    Compute a deterministic SHA-256 hash of the source DataFrame slice.
    """
    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()

    # Sort columns for consistency
    cols = sorted(df.columns)
    # Hash row representations
    hasher = hashlib.sha256()
    # Add column names and shape
    hasher.update(f"{df.shape}:{','.join(cols)}:".encode("utf-8"))
    
    # Use pandas deterministic row hashing
    row_hashes = pd.util.hash_pandas_object(df[cols], index=True).to_numpy()
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def write_scenario_manifest(
    config,
    n_windows: int,
    source_df: pd.DataFrame,
    run_id: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Write runs/<run_id>/manifest.json containing all scenario and slice metadata.

    Args:
        config: ObservatoryConfig
        n_windows: Total number of windows in the stream
        source_df: Source DataFrame slice for the stream period
        run_id: Run identifier (defaults to deterministic '{scenario}_{shape}_seed{seed}')
        out_dir: Directory where runs/ are located (defaults to ROOT_DIR / 'runs')

    Returns:
        Path to the generated manifest.json
    """
    drift_cfg = config.drift_injection
    resolved_run_id = run_id or f"{drift_cfg.scenario}_{drift_cfg.shape}_seed{config.seed}"
    runs_dir = (out_dir or (ROOT_DIR / "runs")) / resolved_run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    data_hash = compute_slice_hash(source_df)
    git_commit = get_git_commit()

    manifest_data: Dict[str, Any] = {
        "run_id": resolved_run_id,
        "scenario": drift_cfg.scenario,
        "shape": drift_cfg.shape,
        "seed": config.seed,
        "onset_window": drift_cfg.onset_window,
        "ramp_windows": drift_cfg.ramp_windows,
        "magnitude": drift_cfg.magnitude,
        "window_size_days": config.window_size_days,
        "n_windows": n_windows,
        "stream_start": config.stream_start,
        "stream_end": config.stream_end,
        "data_hash": data_hash,
        "git_commit": git_commit,
        "affected_categories": drift_cfg.affected_categories,
    }

    manifest_path = runs_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Scenario manifest written to %s", manifest_path)

    # Also update runs/latest/manifest.json if this is the default runs directory
    if out_dir is None:
        latest_dir = ROOT_DIR / "runs" / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_manifest = latest_dir / "manifest.json"
        with latest_manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    return manifest_path
