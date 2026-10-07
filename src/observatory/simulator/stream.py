"""
src/observatory/simulator/stream.py — CreditOps v2 Observatory, Phase 1

Window-based stream generator for offline drift simulation.

Uses DatasetAdapter.time_windows() (V1, unchanged) to produce time-ordered
windows (default 2-day) from the stream period. For each window, applies
the configured drift scenario via injector.py.

Yields StreamWindow objects (accessible as objects or dicts) with:
  - window_index
  - start_date
  - end_date
  - dataframe (features_df post-injection)
  - is_drifted (ground truth boolean)
  - drift_strength (0.0 .. 1.0)

Features are built with src/features.build_features() (V1 single source of truth).
Never re-derives or duplicates feature logic.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Dict

import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class StreamWindow(Mapping):
    """
    One stream window yielded by WindowedStream.
    Supports both attribute access (window.is_drifted) and dict access (window['is_drifted']).
    """
    window_index: int
    start_date: str
    end_date: str
    dataframe: pd.DataFrame
    is_drifted: bool
    drift_strength: float

    # Aliases and metadata for backward compatibility & V1 integration
    index: int = field(init=False)
    features_df: pd.DataFrame = field(init=False)
    labels: pd.Series = field(default=None)
    raw_df: pd.DataFrame = field(default=None)
    n_rows: int = 0
    n_fraud: int = 0
    fraud_rate: float = 0.0
    drift_active: bool = False
    effective_magnitude: float = 0.0
    period_label: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = self.window_index
        self.features_df = self.dataframe
        if self.labels is None:
            self.labels = pd.Series([0] * len(self.dataframe), dtype=int)
        if self.raw_df is None:
            self.raw_df = pd.DataFrame()
        self.n_rows = len(self.dataframe)
        self.n_fraud = int(self.labels.sum())
        self.fraud_rate = float(self.n_fraud / self.n_rows) if self.n_rows > 0 else 0.0
        self.drift_active = self.is_drifted
        self.effective_magnitude = self.drift_strength

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter([
            "window_index", "start_date", "end_date", "dataframe",
            "is_drifted", "drift_strength", "n_rows", "n_fraud",
            "fraud_rate", "features_df", "labels", "raw_df",
        ])

    def __len__(self) -> int:
        return 12

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "dataframe": self.dataframe,
            "is_drifted": self.is_drifted,
            "drift_strength": self.drift_strength,
            "n_rows": self.n_rows,
            "n_fraud": self.n_fraud,
            "fraud_rate": self.fraud_rate,
        }


class WindowedStream:
    """
    Generates a reproducible, drift-injected sequence of stream windows
    from a raw transaction DataFrame.
    """

    def __init__(
        self,
        raw_df: Optional[pd.DataFrame] = None,
        config = None,
        label_col: str = "is_fraud",
    ) -> None:
        if config is None:
            from src.observatory.config import load_config
            config = load_config()
        self._cfg = config
        self._label_col = label_col
        self._drift_cfg = config.drift_injection

        try:
            from src.dataset_adapter import DatasetAdapter
            from src.features import build_features
        except ImportError:
            from dataset_adapter import DatasetAdapter  # type: ignore
            from features import build_features  # type: ignore

        self._build_features = build_features
        self._adapter = DatasetAdapter(
            train_path=getattr(self._cfg, "raw_train", None),
            test_path=getattr(self._cfg, "raw_test", None),
        )

        if raw_df is None:
            self._raw_df = self._adapter.load_slice(self._cfg.stream_start, self._cfg.stream_end)
        else:
            self._raw_df = raw_df.copy()

    def _filter_to_stream_period(self) -> pd.DataFrame:
        """Filter raw_df to the configured stream_start..stream_end period."""
        time_col = self._adapter.time_column
        if time_col not in self._raw_df.columns:
            return pd.DataFrame()

        df = self._raw_df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        start = pd.Timestamp(self._cfg.stream_start)
        end = pd.Timestamp(self._cfg.stream_end) + pd.Timedelta(days=1)
        mask = (df[time_col] >= start) & (df[time_col] < end)
        return df[mask].copy()

    def count_windows(self, window_size_days: Optional[int] = None) -> int:
        """Count total non-empty windows in the stream period."""
        days = window_size_days or self._cfg.window_size_days
        stream_df = self._filter_to_stream_period()
        if stream_df.empty:
            return 0
        return sum(
            1 for _, w in self._adapter.time_windows(stream_df, freq=f"{days}D")
            if not w.empty
        )

    def __iter__(self) -> Iterator[StreamWindow]:
        from src.observatory.drift.injector import apply_drift, _effective_magnitude

        stream_df = self._filter_to_stream_period()
        if stream_df.empty:
            logger.warning(
                "Stream DataFrame is empty for period %s to %s.",
                self._cfg.stream_start, self._cfg.stream_end,
            )
            return

        window_size_days = self._cfg.window_size_days
        total_windows = self.count_windows(window_size_days)

        # Fallback if too few windows: shrink window_size_days
        min_windows = getattr(self._cfg, "min_windows", 10)
        if 0 < total_windows < min_windows and window_size_days > 1:
            new_window_size = max(1, window_size_days // 2)
            warning_msg = (
                f"Stream has too few windows ({total_windows} < {min_windows}). "
                f"Fallback triggered: shrinking window_size_days from {window_size_days} to {new_window_size}."
            )
            logger.warning(warning_msg)
            warnings.warn(warning_msg)
            window_size_days = new_window_size
            total_windows = self.count_windows(window_size_days)

        logger.info("Total stream windows: %d (window size: %d days)", total_windows, window_size_days)

        min_fraud = getattr(self._cfg, "min_fraud_per_window", 5)
        drift_cfg = self._drift_cfg

        window_index = 0
        for period_label, window_raw in self._adapter.time_windows(
            stream_df, freq=f"{window_size_days}D"
        ):
            if window_raw.empty:
                logger.debug("Skipping empty window at %s (index %d).", period_label, window_index)
                window_index += 1
                continue

            # Parse start and end date from period_label or timestamps
            parts = period_label.split("/")
            start_date = parts[0].strip() if len(parts) > 0 else ""
            end_date = parts[1].strip() if len(parts) > 1 else start_date

            # Build V1 features (never duplicate feature logic)
            features_df = self._build_features(window_raw)

            labels = (
                window_raw[self._label_col].reset_index(drop=True)
                if self._label_col in window_raw.columns
                else pd.Series([0] * len(features_df), dtype=int)
            )

            eff_mag = _effective_magnitude(
                window_index,
                drift_cfg.onset_window,
                drift_cfg.ramp_windows,
                drift_cfg.magnitude,
                drift_cfg.shape,
            )
            is_drifted = (eff_mag > 0.0) and (drift_cfg.scenario != "control")
            drift_strength = float(eff_mag / drift_cfg.magnitude) if drift_cfg.magnitude > 0 else 0.0
            drift_strength = min(max(drift_strength, 0.0), 1.0)

            features_df_mod, labels_mod = apply_drift(
                features_df,
                labels,
                window_index,
                scenario=drift_cfg.scenario,
                shape=drift_cfg.shape,
                onset_window=drift_cfg.onset_window,
                ramp_windows=drift_cfg.ramp_windows,
                magnitude=drift_cfg.magnitude,
                affected_categories=drift_cfg.affected_categories,
                seed=self._cfg.seed,
            )

            n_fraud = int(labels_mod.sum())
            n_rows = len(labels_mod)

            # Log window details and check min fraud threshold
            logger.info(
                "Window %d (%s): %d fraud rows out of %d total (fraud rate: %.4f, drifted: %s)",
                window_index, period_label, n_fraud, n_rows, (n_fraud / n_rows if n_rows else 0.0), is_drifted,
            )

            if n_fraud < min_fraud:
                warn_text = (
                    f"Window {window_index} ({period_label}) has {n_fraud} fraud rows, "
                    f"which is fewer than threshold N={min_fraud}."
                )
                logger.warning(warn_text)
                warnings.warn(warn_text)

            yield StreamWindow(
                window_index=window_index,
                start_date=start_date,
                end_date=end_date,
                dataframe=features_df_mod,
                is_drifted=is_drifted,
                drift_strength=round(drift_strength, 6),
                labels=labels_mod,
                raw_df=window_raw.reset_index(drop=True),
                period_label=period_label,
                effective_magnitude=round(eff_mag, 6),
                metadata={
                    "scenario": drift_cfg.scenario,
                    "shape": drift_cfg.shape,
                    "onset_window": drift_cfg.onset_window,
                    "seed": self._cfg.seed,
                },
            )
            window_index += 1
