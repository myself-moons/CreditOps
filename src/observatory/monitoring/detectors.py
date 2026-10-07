"""
src/observatory/monitoring/detectors.py — CreditOps v2 Observatory

Implements drift and novelty detectors conforming to the common interface:
    update(window) -> DetectionResult(score, alarm, window_index, detector, details)

Detectors:
  1. DataDriftDetector: aggregate PSI and KS across V1 numeric features and categorical shares
     comparing the pooled current + previous K windows against reference.
  2. NoveltyDetector: IsolationForest fitted on training-period preprocessed features only;
     score = outlier fraction + unseen category fraction.
  3. FraudScoreDriftDetector: PSI of champion model predicted scores vs reference.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.observatory.simulator.stream import StreamWindow
from src.observatory.storage.base import LogStore

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# V1 feature definitions
NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "log_amt",
    "age_at_txn",
    "haversine_km",
    "log_city_pop",
]

CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport",
    "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
    "food_dining", "personal_care", "health_fitness", "travel",
    "kids_pets", "home",
]
CAT_COLS = [f"cat_{c}" for c in CATEGORIES]


@dataclass
class DetectionResult:
    """Standard detection outcome returned by all detectors."""
    score: float
    alarm: bool
    window_index: int
    detector: str
    details: Dict[str, Any] = field(default_factory=dict)


def calculate_psi(
    ref: np.ndarray,
    curr: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and current arrays.
    Handles constant features and empty arrays safely.
    """
    ref_clean = ref[~np.isnan(ref)]
    curr_clean = curr[~np.isnan(curr)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.unique(np.percentile(ref_clean, quantiles))
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_prop = np.where(ref_counts == 0, eps, ref_counts / len(ref_clean))
    curr_prop = np.where(curr_counts == 0, eps, curr_counts / len(curr_clean))

    ref_prop /= np.sum(ref_prop)
    curr_prop /= np.sum(curr_prop)

    psi_val = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
    return float(max(0.0, psi_val))


def calculate_categorical_psi(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    cat_cols: Sequence[str],
    eps: float = 1e-4,
) -> float:
    """Calculate aggregate PSI across categorical one-hot column shares."""
    if len(ref_df) == 0 or len(curr_df) == 0 or not cat_cols:
        return 0.0

    ref_shares = np.array([ref_df[c].mean() if c in ref_df.columns else 0.0 for c in cat_cols])
    curr_shares = np.array([curr_df[c].mean() if c in curr_df.columns else 0.0 for c in cat_cols])

    ref_shares = np.maximum(ref_shares, eps)
    curr_shares = np.maximum(curr_shares, eps)

    ref_shares /= np.sum(ref_shares)
    curr_shares /= np.sum(curr_shares)

    psi_val = np.sum((curr_shares - ref_shares) * np.log(curr_shares / ref_shares))
    return float(max(0.0, psi_val))


class BaseDetector(ABC):
    """Abstract base class for all monitoring detectors."""

    def __init__(
        self,
        name: str,
        threshold: float,
        log_store: Optional[LogStore] = None,
    ) -> None:
        self.name = name
        self.threshold = float(threshold)
        self.log_store = log_store

    def _log_failure(self, window_index: int, error_msg: str) -> DetectionResult:
        logger.warning("[%s] Robustness fallback at window %d: %s", self.name, window_index, error_msg)
        res = DetectionResult(
            score=0.0,
            alarm=False,
            window_index=window_index,
            detector=self.name,
            details={"error": error_msg},
        )
        if self.log_store is not None:
            try:
                self.log_store.append("drift_events", {
                    "detector": self.name,
                    "window_index": window_index,
                    "alarm": False,
                    "score": 0.0,
                    "error": error_msg,
                })
            except Exception as e:
                logger.error("Failed to append to log_store: %s", e)
        return res

    @abstractmethod
    def update(self, window: StreamWindow) -> DetectionResult:
        raise NotImplementedError


class DataDriftDetector(BaseDetector):
    """
    Data Drift Detector:
    Aggregates KS statistics and PSI across numeric features and categorical shares.
    Compares the current window pooled with previous K windows against reference.
    """

    def __init__(
        self,
        reference_df: Optional[pd.DataFrame] = None,
        k_prev: int = 3,
        threshold: float = 0.85,
        numeric_features: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        log_store: Optional[LogStore] = None,
    ) -> None:
        super().__init__(name="data_drift", threshold=threshold, log_store=log_store)
        self.k_prev = max(0, int(k_prev))
        self.numeric_features = list(numeric_features or NUMERIC_FEATURES)
        self.categorical_columns = list(categorical_columns or CAT_COLS)
        self._reference_df = reference_df.copy() if reference_df is not None else None
        self._history: List[pd.DataFrame] = []

    def set_reference(self, reference_df: pd.DataFrame) -> None:
        self._reference_df = reference_df.copy()

    def update(self, window: StreamWindow) -> DetectionResult:
        if window is None or window.features_df is None or len(window.features_df) == 0:
            return self._log_failure(getattr(window, "window_index", -1), "Empty window or missing features")

        df = window.features_df

        # Check required columns
        missing_num = [c for c in self.numeric_features if c not in df.columns]
        if missing_num:
            return self._log_failure(window.window_index, f"Missing numeric columns: {missing_num}")

        # Store in rolling buffer
        self._history.append(df.copy())
        if len(self._history) > (self.k_prev + 1):
            self._history.pop(0)

        # Build pooled evaluation DataFrame
        pool_df = pd.concat(self._history, ignore_index=True)

        # If reference is missing, use current buffer or fail gracefully
        if self._reference_df is None or len(self._reference_df) == 0:
            return self._log_failure(window.window_index, "Reference sample is not initialized")

        try:
            ks_results: Dict[str, float] = {}
            psi_results: Dict[str, float] = {}

            for col in self.numeric_features:
                ref_vals = self._reference_df[col].dropna().values
                curr_vals = pool_df[col].dropna().values

                if len(ref_vals) == 0 or len(curr_vals) == 0:
                    ks_stat = 0.0
                    psi_stat = 0.0
                elif np.all(ref_vals == ref_vals[0]) and np.all(curr_vals == curr_vals[0]):
                    ks_stat = 0.0 if ref_vals[0] == curr_vals[0] else 1.0
                    psi_stat = 0.0
                else:
                    ks_res = ks_2samp(ref_vals, curr_vals)
                    ks_stat = float(ks_res.statistic)
                    psi_stat = calculate_psi(ref_vals, curr_vals)

                ks_results[col] = ks_stat
                psi_results[col] = psi_stat

            cat_psi = calculate_categorical_psi(self._reference_df, pool_df, self.categorical_columns)

            mean_ks = float(np.mean(list(ks_results.values()))) if ks_results else 0.0
            mean_psi = float(np.mean(list(psi_results.values()))) if psi_results else 0.0

            # Aggregate score: mean KS across numeric features + mean PSI + categorical PSI
            score = float(mean_ks + mean_psi + cat_psi)
            alarm = bool(score >= self.threshold)

            details = {
                "score": score,
                "mean_ks": mean_ks,
                "mean_psi": mean_psi,
                "cat_psi": cat_psi,
                "ks_per_feature": ks_results,
                "psi_per_feature": psi_results,
                "pooled_windows": len(self._history),
                "pooled_rows": len(pool_df),
                "threshold": self.threshold,
            }

            if self.log_store is not None and alarm:
                self.log_store.append("drift_events", {
                    "detector": self.name,
                    "window_index": window.window_index,
                    "alarm": alarm,
                    "score": score,
                    "details": details,
                })

            return DetectionResult(
                score=score,
                alarm=alarm,
                window_index=window.window_index,
                detector=self.name,
                details=details,
            )

        except Exception as exc:
            return self._log_failure(window.window_index, f"Detection calculation error: {str(exc)}")


class UnseenCategoryDetector(BaseDetector):
    """
    Unseen Category Detector:
    Alarms when count of rows with all cat_* zero in the window >= min_unseen_rows (default 3).
    """

    def __init__(
        self,
        min_unseen_rows: int = 3,
        categorical_columns: Optional[List[str]] = None,
        log_store: Optional[LogStore] = None,
    ) -> None:
        super().__init__(name="unseen_category", threshold=float(min_unseen_rows), log_store=log_store)
        self.min_unseen_rows = max(1, int(min_unseen_rows))
        self.categorical_columns = list(categorical_columns or CAT_COLS)

    def update(self, window: StreamWindow) -> DetectionResult:
        if window is None or window.features_df is None or len(window.features_df) == 0:
            return self._log_failure(getattr(window, "window_index", -1), "Empty window or missing features")

        df = window.features_df
        try:
            present_cat_cols = [c for c in self.categorical_columns if c in df.columns]
            if present_cat_cols:
                unseen_mask = (df[present_cat_cols].sum(axis=1) == 0)
                unseen_count = int(unseen_mask.sum())
                unseen_frac = float(unseen_mask.mean())
            else:
                unseen_count = 0
                unseen_frac = 0.0

            score = float(unseen_count)
            alarm = bool(unseen_count >= self.min_unseen_rows)

            details = {
                "score": score,
                "unseen_count": unseen_count,
                "unseen_frac": unseen_frac,
                "min_unseen_rows": self.min_unseen_rows,
                "num_samples": len(df),
            }

            if self.log_store is not None and alarm:
                self.log_store.append("drift_events", {
                    "detector": self.name,
                    "window_index": window.window_index,
                    "alarm": alarm,
                    "score": score,
                    "details": details,
                })

            return DetectionResult(
                score=score,
                alarm=alarm,
                window_index=window.window_index,
                detector=self.name,
                details=details,
            )
        except Exception as exc:
            return self._log_failure(window.window_index, f"Unseen category detection error: {str(exc)}")


class IsolationOutlierDetector(BaseDetector):
    """
    Isolation Forest Outlier Detector:
    IsolationForest fitted on a random sample (~50k rows) across the whole training period.
    Score = fraction of rows flagged as outliers (preds == -1).
    """

    def __init__(
        self,
        isolation_forest: Any,
        preprocessor: Any,
        threshold: float = 0.130,
        log_store: Optional[LogStore] = None,
    ) -> None:
        super().__init__(name="isolation_outlier", threshold=threshold, log_store=log_store)
        self.model = isolation_forest
        self.preprocessor = preprocessor

    def update(self, window: StreamWindow) -> DetectionResult:
        if window is None or window.features_df is None or len(window.features_df) == 0:
            return self._log_failure(getattr(window, "window_index", -1), "Empty window or missing features")

        df = window.features_df
        try:
            X_enc = self.preprocessor.transform(df)
            preds = self.model.predict(X_enc)
            outlier_frac = float((preds == -1).mean())

            score = outlier_frac
            alarm = bool(score >= self.threshold)

            details = {
                "score": score,
                "outlier_frac": outlier_frac,
                "num_samples": len(df),
                "threshold": self.threshold,
            }

            if self.log_store is not None and alarm:
                self.log_store.append("drift_events", {
                    "detector": self.name,
                    "window_index": window.window_index,
                    "alarm": alarm,
                    "score": score,
                    "details": details,
                })

            return DetectionResult(
                score=score,
                alarm=alarm,
                window_index=window.window_index,
                detector=self.name,
                details=details,
            )
        except Exception as exc:
            return self._log_failure(window.window_index, f"Isolation outlier detection error: {str(exc)}")


class NoveltyDetector(BaseDetector):
    """
    Novelty Detector (Legacy composite):
    IsolationForest fitted on training-period preprocessed features only;
    score = fraction of rows flagged as outliers + fraction with an unseen category (all cat_* zero).
    """

    def __init__(
        self,
        isolation_forest: Any,
        preprocessor: Any,
        threshold: float = 0.24,
        categorical_columns: Optional[List[str]] = None,
        log_store: Optional[LogStore] = None,
    ) -> None:
        super().__init__(name="novelty", threshold=threshold, log_store=log_store)
        self.model = isolation_forest
        self.preprocessor = preprocessor
        self.categorical_columns = list(categorical_columns or CAT_COLS)

    def update(self, window: StreamWindow) -> DetectionResult:
        if window is None or window.features_df is None or len(window.features_df) == 0:
            return self._log_failure(getattr(window, "window_index", -1), "Empty window or missing features")

        df = window.features_df

        try:
            # Fraction with unseen category: all cat_* zero
            present_cat_cols = [c for c in self.categorical_columns if c in df.columns]
            if present_cat_cols:
                unseen_mask = (df[present_cat_cols].sum(axis=1) == 0)
                unseen_frac = float(unseen_mask.mean())
            else:
                unseen_frac = 0.0

            # IsolationForest outlier prediction on preprocessed features
            X_enc = self.preprocessor.transform(df)
            preds = self.model.predict(X_enc)
            outlier_frac = float((preds == -1).mean())

            score = float(outlier_frac + unseen_frac)
            alarm = bool(score >= self.threshold)

            details = {
                "score": score,
                "outlier_frac": outlier_frac,
                "unseen_category_frac": unseen_frac,
                "num_samples": len(df),
                "threshold": self.threshold,
            }

            if self.log_store is not None and alarm:
                self.log_store.append("drift_events", {
                    "detector": self.name,
                    "window_index": window.window_index,
                    "alarm": alarm,
                    "score": score,
                    "details": details,
                })

            return DetectionResult(
                score=score,
                alarm=alarm,
                window_index=window.window_index,
                detector=self.name,
                details=details,
            )

        except Exception as exc:
            return self._log_failure(window.window_index, f"Novelty detection error: {str(exc)}")


class FraudScoreDriftDetector(BaseDetector):
    """
    Fraud Score Drift Detector:
    PSI of the champion model's predicted-score distribution vs reference.
    """

    def __init__(
        self,
        champion_model: Any,
        preprocessor: Any,
        reference_scores: Optional[np.ndarray] = None,
        threshold: float = 0.025,
        log_store: Optional[LogStore] = None,
    ) -> None:
        super().__init__(name="fraud_score_drift", threshold=threshold, log_store=log_store)
        self.champion_model = champion_model
        self.preprocessor = preprocessor
        self._reference_scores = np.array(reference_scores) if reference_scores is not None else None

    def set_reference_scores(self, scores: np.ndarray) -> None:
        self._reference_scores = np.array(scores)

    def update(self, window: StreamWindow) -> DetectionResult:
        if window is None or window.features_df is None or len(window.features_df) == 0:
            return self._log_failure(getattr(window, "window_index", -1), "Empty window or missing features")

        df = window.features_df

        if self._reference_scores is None or len(self._reference_scores) == 0:
            return self._log_failure(window.window_index, "Reference predicted scores not initialized")

        try:
            X_enc = self.preprocessor.transform(df)
            curr_scores = self.champion_model.predict_proba(X_enc)[:, 1]

            score = calculate_psi(self._reference_scores, curr_scores)
            alarm = bool(score >= self.threshold)

            details = {
                "score": score,
                "ref_mean_score": float(np.mean(self._reference_scores)),
                "curr_mean_score": float(np.mean(curr_scores)),
                "threshold": self.threshold,
            }

            if self.log_store is not None and alarm:
                self.log_store.append("drift_events", {
                    "detector": self.name,
                    "window_index": window.window_index,
                    "alarm": alarm,
                    "score": score,
                    "details": details,
                })

            return DetectionResult(
                score=score,
                alarm=alarm,
                window_index=window.window_index,
                detector=self.name,
                details=details,
            )

        except Exception as exc:
            return self._log_failure(window.window_index, f"Fraud score drift error: {str(exc)}")
