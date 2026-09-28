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

    def __init__(self, params: Optional[dict] = None):
        self._params = params or _load_params()
        data_cfg = self._params.get("data", {})
        self._train_path = ROOT_DIR / data_cfg.get("raw_train", "Credit_Data/fraudTrain.csv")
        self._test_path  = ROOT_DIR / data_cfg.get("raw_test",  "Credit_Data/fraudTest.csv")
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
