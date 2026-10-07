"""
dataset_adapter.py — CreditOps

Single DatasetAdapter class that owns all data access.
Other modules import ONLY this class and call its methods.
This makes swapping to a new dataset for the capstone a one-file change.

V2-READINESS: implements load(), time_column, label_column, split_by_time(),
and time_windows() so monitoring and retraining can reuse them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_params() -> dict:
    params_path = ROOT_DIR / "params.yaml"
    if not params_path.exists():
        return {}
    with params_path.open() as f:
        return yaml.safe_load(f)


class DatasetAdapter:
    """
    Abstracts all access to the credit-card fraud detection dataset.

    Dataset: Kaggle "Credit Card Transactions Fraud Detection Dataset"
    (Sparkov simulation; kartik2112/fraud-detection).
    This is SIMULATED data — no real cardholders are involved.
    License: CC0 1.0 (Public Domain Dedication) as stated on the Kaggle page.

    Columns present (verified on load):
        Unnamed: 0, trans_date_trans_time, cc_num, merchant, category, amt,
        first, last, gender, street, city, state, zip, lat, long, city_pop,
        job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud

    Data provenance: "synthetic/simulated (Sparkov)"
    """

    # Expected columns from the Kaggle dataset
    EXPECTED_COLS = [
        "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
        "first", "last", "gender", "street", "city", "state", "zip",
        "lat", "long", "city_pop", "job", "dob", "trans_num",
        "unix_time", "merch_lat", "merch_long", "is_fraud",
    ]

    DATA_PROVENANCE = "synthetic/simulated (Sparkov)"
    LICENSE = "CC0 1.0 Universal (Public Domain Dedication)"

    def __init__(
        self,
        params: Optional[dict] = None,
        train_path: Optional[str | Path] = None,
        test_path: Optional[str | Path] = None,
    ):
        self._params = params or _load_params()
        data_cfg = self._params.get("data", {})
        default_train = ROOT_DIR / "data" / "fraudTrain.csv"
        default_test  = ROOT_DIR / "data" / "fraudTest.csv"
        self._train_path = Path(train_path) if train_path else (
            ROOT_DIR / data_cfg.get("raw_train", "Credit_Data/fraudTrain.csv")
        )
        if not self._train_path.exists() and default_train.exists():
            self._train_path = default_train

        self._test_path = Path(test_path) if test_path else (
            ROOT_DIR / data_cfg.get("raw_test", "Credit_Data/fraudTest.csv")
        )
        if not self._test_path.exists() and default_test.exists():
            self._test_path = default_test

        self._time_col   = data_cfg.get("time_col", "trans_date_trans_time")
        self._label_col  = data_cfg.get("label_col", "is_fraud")

    @property
    def time_column(self) -> str:
        return self._time_col

    @property
    def label_column(self) -> str:
        return self._label_col

    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Concatenate train and test CSVs, parse timestamps, drop exact duplicates,
        validate schema, sort by time.  Logs dropped rows and why.

        Returns a single time-sorted DataFrame ready for split_by_time().
        """
        logger.info("Loading fraudTrain.csv (%s rows)...", nrows or "all")
        df_train = pd.read_csv(self._train_path, nrows=nrows, low_memory=False)
        logger.info("Loading fraudTest.csv (%s rows)...", nrows or "all")
        df_test  = pd.read_csv(self._test_path,  nrows=nrows, low_memory=False)

        # Drop the row-index column Kaggle adds
        for df in (df_train, df_test):
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

        df = pd.concat([df_train, df_test], ignore_index=True)
        initial_rows = len(df)
        logger.info("Concatenated: %d rows", initial_rows)

        # --- Schema validation ------------------------------------------------
        missing = [c for c in self.EXPECTED_COLS if c not in df.columns]
        extra   = [c for c in df.columns if c not in self.EXPECTED_COLS]
        if missing:
            logger.warning("Missing expected columns: %s", missing)
        if extra:
            logger.info("Extra columns (will be kept): %s", extra)

        # --- Parse timestamps -------------------------------------------------
        df[self._time_col] = pd.to_datetime(df[self._time_col], errors="coerce")
        ts_nulls = df[self._time_col].isna().sum()
        if ts_nulls:
            logger.warning("Dropping %d rows with un-parseable timestamps", ts_nulls)
            df = df[df[self._time_col].notna()].copy()

        # --- Drop exact duplicates --------------------------------------------
        before_dedup = len(df)
        df = df.drop_duplicates()
        dropped_dups = before_dedup - len(df)
        if dropped_dups:
            logger.info("Dropped %d exact duplicate rows", dropped_dups)

        # --- Sort by time -----------------------------------------------------
        df = df.sort_values(self._time_col).reset_index(drop=True)

        logger.info(
            "Final: %d rows (dropped %d = %d dups + %d bad timestamps)",
            len(df), initial_rows - len(df), dropped_dups, ts_nulls,
        )
        return df

    def load_slice(self, start: str, end: str) -> pd.DataFrame:
        """
        Load and return a slice of real Sparkov data between start and end dates.
        Only reads the required CSV file(s) for speed, parses timestamps,
        drops duplicates, sorts by time, and filters to [start, end].
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta("1ns")

        train_path = self._train_path if self._train_path.exists() else (ROOT_DIR / "data" / "fraudTrain.csv")
        test_path = self._test_path if self._test_path.exists() else (ROOT_DIR / "data" / "fraudTest.csv")

        # fraudTrain ends on 2020-06-21 12:13:37; fraudTest starts on 2020-06-21 12:14:25
        split_boundary = pd.Timestamp("2020-06-21 12:14:00")

        dfs = []
        if start_ts < split_boundary:
            logger.info("load_slice: reading %s for period starting %s", train_path.name, start)
            df_tr = pd.read_csv(train_path, low_memory=False)
            if "Unnamed: 0" in df_tr.columns:
                df_tr.drop(columns=["Unnamed: 0"], inplace=True)
            dfs.append(df_tr)

        if end_ts >= split_boundary:
            logger.info("load_slice: reading %s for period ending %s", test_path.name, end)
            df_te = pd.read_csv(test_path, low_memory=False)
            if "Unnamed: 0" in df_te.columns:
                df_te.drop(columns=["Unnamed: 0"], inplace=True)
            dfs.append(df_te)

        if not dfs:
            return pd.DataFrame(columns=self.EXPECTED_COLS)

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        df[self._time_col] = pd.to_datetime(df[self._time_col], errors="coerce")
        df = df[df[self._time_col].notna()].copy()

        mask = (df[self._time_col] >= start_ts) & (df[self._time_col] <= end_ts)
        df = df[mask].drop_duplicates().sort_values(self._time_col).reset_index(drop=True)

        logger.info(
            "load_slice (%s to %s): loaded %d rows (%d fraud, rate=%.5f)",
            start, end, len(df),
            int(df[self._label_col].sum()) if self._label_col in df.columns else 0,
            float(df[self._label_col].mean()) if self._label_col in df.columns else 0.0,
        )
        return df

    def split_by_time(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split: first train_frac rows → train,
        next val_frac rows → val, remainder → test.

        NEVER splits randomly — preserves temporal order to prevent leakage.

        Returns (train_df, val_df, test_df).
        """
        n = len(df)
        train_end = int(n * train_frac)
        val_end   = int(n * (train_frac + val_frac))

        train = df.iloc[:train_end].copy()
        val   = df.iloc[train_end:val_end].copy()
        test  = df.iloc[val_end:].copy()

        # Verify no overlap
        assert train[self._time_col].max() <= val[self._time_col].min(), \
            "Temporal split FAILED: train max >= val min"
        assert val[self._time_col].max() <= test[self._time_col].min(), \
            "Temporal split FAILED: val max >= test min"

        return train, val, test

    def time_windows(
        self,
        df: pd.DataFrame,
        freq: str = "W",
    ) -> Generator[tuple[str, pd.DataFrame], None, None]:
        """
        Yield (period_label, window_df) for each calendar period in df.
        Used by monitoring and stationarity checks.

        Args:
            df:   Any time-sorted DataFrame with self.time_column present.
            freq: pandas offset alias — "W" (weekly), "ME" (month-end), etc.
        """
        df = df.copy()
        df["_period"] = df[self._time_col].dt.to_period(freq)
        for period, group in df.groupby("_period", sort=True):
            yield str(period), group.drop(columns=["_period"])
