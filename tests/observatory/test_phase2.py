"""
tests/observatory/test_phase2.py — CreditOps v2 Observatory Phase 2 Tests

Validates:
  1. SQLite LogStore write/read/filter interface contract and collection validation.
  2. Detector common interface contract (DetectionResult attributes).
  3. Label-delay compliance (no future labels used before delay elapsed).
  4. Robustness edge cases (empty window, missing columns, NaNs, constant columns).
  5. Calibration integrity and zero onset leakage (calibrated on windows 0-14, <= 1 FA on 15-88).
  6. PerformanceMonitor insufficient data handling (returns NaN without crashing).
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.features import build_features
from src.observatory.config import DriftInjectionConfig, load_config
from src.observatory.monitoring.detectors import (
    CAT_COLS,
    NUMERIC_FEATURES,
    DataDriftDetector,
    DetectionResult,
    FraudScoreDriftDetector,
    NoveltyDetector,
)
from src.observatory.monitoring.performance import PerformanceMonitor
from src.observatory.simulator.stream import StreamWindow, WindowedStream
from src.observatory.storage.base import LogStore, VALID_COLLECTIONS
from src.observatory.storage.sqlite_store import SQLiteLogStore

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def memory_log_store() -> SQLiteLogStore:
    return SQLiteLogStore(db_path=":memory:")


@pytest.fixture
def v1_champion_assets():
    model_path = ROOT / "model.pkl"
    prep_path = ROOT / "preprocessor.pkl"
    thresh_path = ROOT / ".decision_threshold"

    if not model_path.exists() or not prep_path.exists() or not thresh_path.exists():
        pytest.skip("V1 champion assets not found in workspace")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)
    with open(thresh_path, "r", encoding="utf-8") as f:
        threshold = float(f.read().strip())

    return model, prep, threshold


@pytest.fixture
def synthetic_window() -> StreamWindow:
    """Creates a deterministic small synthetic StreamWindow for unit testing."""
    n = 200
    dates = pd.date_range("2020-10-04", periods=n, freq="1min")
    raw = pd.DataFrame({
        "trans_date_trans_time": dates,
        "amt": np.random.uniform(10.0, 500.0, size=n),
        "category": np.random.choice(["grocery_pos", "misc_net", "travel"], size=n),
        "gender": np.random.choice(["M", "F"], size=n),
        "dob": "1980-01-01",
        "lat": 40.0,
        "long": -73.0,
        "merch_lat": 40.1,
        "merch_long": -73.1,
        "city_pop": 50000,
        "is_fraud": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
    })
    feats = build_features(raw)
    labels = raw["is_fraud"].copy()

    return StreamWindow(
        window_index=0,
        start_date="2020-10-04",
        end_date="2020-10-05",
        dataframe=feats,
        is_drifted=False,
        drift_strength=0.0,
        raw_df=raw,
        labels=labels,
    )


# ============================================================================ #
# 1. SQLite LogStore Contract                                                  #
# ============================================================================ #

class TestSQLiteLogStore:
    def test_implements_interface(self, memory_log_store: SQLiteLogStore):
        assert isinstance(memory_log_store, LogStore)

    def test_invalid_collection_raises_value_error(self, memory_log_store: SQLiteLogStore):
        with pytest.raises(ValueError, match="Unknown collection"):
            memory_log_store.append("invalid_collection_name", {"event": "test"})

        with pytest.raises(ValueError, match="Unknown collection"):
            memory_log_store.query("invalid_collection_name")

    def test_append_and_query_drift_events(self, memory_log_store: SQLiteLogStore):
        rec_id = memory_log_store.append("drift_events", {
            "detector": "data_drift",
            "window_index": 5,
            "alarm": True,
            "score": 1.25,
            "details": {"test_key": "val1"},
        })
        assert isinstance(rec_id, str)
        assert len(rec_id) > 0

        # Query all
        results = memory_log_store.query("drift_events")
        assert len(results) == 1
        rec = results[0]
        assert rec["id"] == rec_id
        assert rec["detector"] == "data_drift"
        assert rec["window_index"] == 5
        assert rec["alarm"] is True
        assert rec["score"] == 1.25
        assert rec["details"]["test_key"] == "val1"
        assert "timestamp" in rec

    def test_query_filtering_and_limit(self, memory_log_store: SQLiteLogStore):
        for i in range(10):
            memory_log_store.append("drift_events", {
                "detector": "novelty" if i % 2 == 0 else "data_drift",
                "window_index": i,
                "alarm": (i >= 5),
            })

        # Limit
        assert len(memory_log_store.query("drift_events", limit=4)) == 4

        # Filter by detector
        nov_records = memory_log_store.query("drift_events", filters={"detector": "novelty"})
        assert len(nov_records) == 5
        assert all(r["detector"] == "novelty" for r in nov_records)

        # Filter by alarm
        alarmed = memory_log_store.query("drift_events", filters={"alarm": True})
        assert len(alarmed) == 5
        assert all(r["alarm"] is True for r in alarmed)


# ============================================================================ #
# 2. Detector Interface Contract                                               #
# ============================================================================ #

class TestDetectorInterfaceContract:
    def test_detection_result_structure(
        self, synthetic_window: StreamWindow, memory_log_store: SQLiteLogStore, v1_champion_assets
    ):
        model, prep, threshold = v1_champion_assets

        # Fit small IsolationForest
        iforest = IsolationForest(n_estimators=10, random_state=42)
        X_enc = prep.transform(synthetic_window.features_df)
        iforest.fit(X_enc)

        # Test DataDriftDetector
        dd = DataDriftDetector(
            reference_df=synthetic_window.features_df,
            threshold=1.0,
            log_store=memory_log_store,
        )
        res_dd = dd.update(synthetic_window)
        assert isinstance(res_dd, DetectionResult)
        assert isinstance(res_dd.score, float)
        assert isinstance(res_dd.alarm, bool)
        assert res_dd.window_index == synthetic_window.window_index
        assert res_dd.detector == "data_drift"
        assert isinstance(res_dd.details, dict)

        # Test NoveltyDetector
        nov = NoveltyDetector(
            isolation_forest=iforest,
            preprocessor=prep,
            threshold=0.5,
            log_store=memory_log_store,
        )
        res_nov = nov.update(synthetic_window)
        assert isinstance(res_nov, DetectionResult)
        assert isinstance(res_nov.score, float)
        assert isinstance(res_nov.alarm, bool)
        assert res_nov.detector == "novelty"

        # Test FraudScoreDriftDetector
        scores = model.predict_proba(X_enc)[:, 1]
        fs = FraudScoreDriftDetector(
            champion_model=model,
            preprocessor=prep,
            reference_scores=scores,
            threshold=0.05,
            log_store=memory_log_store,
        )
        res_fs = fs.update(synthetic_window)
        assert isinstance(res_fs, DetectionResult)
        assert isinstance(res_fs.score, float)
        assert isinstance(res_fs.alarm, bool)
        assert res_fs.detector == "fraud_score_drift"


# ============================================================================ #
# 3. Label-Delay Compliance (No Future Label Leakage)                          #
# ============================================================================ #

class TestLabelDelayCompliance:
    def test_no_labels_used_from_future_windows(self, v1_champion_assets, synthetic_window):
        model, prep, threshold = v1_champion_assets

        monitor = PerformanceMonitor(
            champion_model=model,
            preprocessor=prep,
            label_delay_windows=2,
            k_perf=5,
            min_fraud_rows=2,
            decision_threshold=threshold,
        )

        def make_window(idx: int, fraud_labels: Sequence[int]) -> StreamWindow:
            n = len(fraud_labels)
            dates = pd.date_range("2020-10-04", periods=n, freq="1min")
            raw = pd.DataFrame({
                "trans_date_trans_time": dates,
                "amt": np.random.uniform(50.0, 100.0, size=n),
                "category": "grocery_pos",
                "gender": "M",
                "dob": "1980-01-01",
                "lat": 40.0,
                "long": -73.0,
                "merch_lat": 40.1,
                "merch_long": -73.1,
                "city_pop": 50000,
                "is_fraud": list(fraud_labels),
            })
            return StreamWindow(
                window_index=idx,
                start_date=f"2020-10-{idx+1:02d}",
                end_date=f"2020-10-{idx+2:02d}",
                dataframe=build_features(raw),
                is_drifted=False,
                drift_strength=0.0,
                raw_df=raw,
                labels=raw["is_fraud"].copy(),
            )

        # Window 0 arrives: delay=2 -> labels not available
        w0 = make_window(0, [1, 1, 0, 0, 0])
        res0 = monitor.update(w0)
        assert np.isnan(res0.score)
        assert res0.details["status"] == "insufficient_data"

        # Window 1 arrives: delay=2 -> labels for w1 not available (max eligible < 0)
        w1 = make_window(1, [1, 1, 0, 0, 0])
        res1 = monitor.update(w1)
        assert np.isnan(res1.score)

        # Window 2 arrives: only window 0 labels are available (t=2, 2-2=0)
        w2 = make_window(2, [0, 0, 0, 0, 0])
        res2 = monitor.update(w2)
        assert res2.details["status"] == "ok"
        assert res2.details["evaluated_windows"] == [0]
        assert not np.isnan(res2.score)

        # Window 3 arrives: windows <= 1 are available ([0, 1])
        w3 = make_window(3, [0, 0, 0, 0, 0])
        res3 = monitor.update(w3)
        assert res3.details["evaluated_windows"] == [0, 1]


# ============================================================================ #
# 4. Robustness Edge Cases (Empty, NaNs, Missing Columns, Constant Columns)    #
# ============================================================================ #

class TestRobustnessEdgeCases:
    def test_empty_window_handling(
        self, memory_log_store: SQLiteLogStore, v1_champion_assets, synthetic_window
    ):
        model, prep, threshold = v1_champion_assets
        empty_raw = pd.DataFrame(columns=synthetic_window.raw_df.columns)
        empty_feats = pd.DataFrame(columns=synthetic_window.features_df.columns)
        empty_labels = pd.Series(dtype=int)

        empty_win = StreamWindow(
            window_index=99,
            start_date="2020-10-04",
            end_date="2020-10-05",
            dataframe=empty_feats,
            is_drifted=False,
            drift_strength=0.0,
            raw_df=empty_raw,
            labels=empty_labels,
        )

        dd = DataDriftDetector(reference_df=synthetic_window.features_df, log_store=memory_log_store)
        res_dd = dd.update(empty_win)
        assert res_dd.alarm is False
        assert "error" in res_dd.details

        nov = NoveltyDetector(
            isolation_forest=IsolationForest(n_estimators=5, random_state=42).fit(prep.transform(synthetic_window.features_df)),
            preprocessor=prep,
            log_store=memory_log_store,
        )
        res_nov = nov.update(empty_win)
        assert res_nov.alarm is False
        assert "error" in res_nov.details

        fs = FraudScoreDriftDetector(
            champion_model=model,
            preprocessor=prep,
            reference_scores=np.array([0.1, 0.2]),
            log_store=memory_log_store,
        )
        res_fs = fs.update(empty_win)
        assert res_fs.alarm is False
        assert "error" in res_fs.details

        # Check logged failure in LogStore
        logged = memory_log_store.query("drift_events")
        assert len(logged) >= 3

    def test_missing_and_constant_columns_handling(
        self, memory_log_store: SQLiteLogStore, synthetic_window
    ):
        # 1. Missing columns: drop log_amt
        bad_feats = synthetic_window.features_df.drop(columns=["log_amt"])
        bad_win = StreamWindow(
            window_index=1,
            start_date="2020-10-04",
            end_date="2020-10-05",
            dataframe=bad_feats,
            is_drifted=False,
            drift_strength=0.0,
            raw_df=synthetic_window.raw_df,
            labels=synthetic_window.labels,
        )
        dd = DataDriftDetector(reference_df=synthetic_window.features_df, log_store=memory_log_store)
        res = dd.update(bad_win)
        assert res.alarm is False
        assert "Missing numeric columns" in res.details["error"]

        # 2. Constant columns: constant value across all rows
        const_feats = synthetic_window.features_df.copy()
        const_feats["log_amt"] = 5.0
        const_win = StreamWindow(
            window_index=2,
            start_date="2020-10-04",
            end_date="2020-10-05",
            dataframe=const_feats,
            is_drifted=False,
            drift_strength=0.0,
            raw_df=synthetic_window.raw_df,
            labels=synthetic_window.labels,
        )
        res_const = dd.update(const_win)
        assert isinstance(res_const.score, float)
        assert not np.isnan(res_const.score)

    def test_nans_in_features_handling(self, synthetic_window, memory_log_store):
        nan_feats = synthetic_window.features_df.copy()
        nan_feats.iloc[:20, nan_feats.columns.get_loc("log_amt")] = np.nan
        nan_win = StreamWindow(
            window_index=3,
            start_date="2020-10-04",
            end_date="2020-10-05",
            dataframe=nan_feats,
            is_drifted=False,
            drift_strength=0.0,
            raw_df=synthetic_window.raw_df,
            labels=synthetic_window.labels,
        )
        dd = DataDriftDetector(reference_df=synthetic_window.features_df, log_store=memory_log_store)
        res = dd.update(nan_win)
        assert isinstance(res.score, float)
        assert not np.isnan(res.score)


# ============================================================================ #
# 5. Calibration & Zero Onset Leakage Check                                    #
# ============================================================================ #

class TestCalibrationAndOnsetLeakage:
    def test_calibration_and_control_false_alarm_rate(self):
        cfg = load_config()

        # Check configured thresholds exist and match derivation
        assert cfg.monitoring.threshold_data_drift == 1.000
        assert cfg.monitoring.threshold_unseen_category == 3
        assert cfg.monitoring.threshold_isolation_outlier == 0.130
        assert cfg.monitoring.threshold_fraud_score_drift == 0.020
        assert cfg.monitoring.threshold_performance_recall == 0.500
        assert cfg.monitoring.threshold_pr_auc_drop == 0.180

        # Verify onset window is strictly window 15
        assert cfg.drift_injection.onset_window == 15

        # Check sanity table exists in docs/detector_sanity.md
        sanity_report = ROOT / "docs" / "detector_sanity.md"
        assert sanity_report.exists(), "docs/detector_sanity.md was not generated"
        content = sanity_report.read_text(encoding="utf-8")

        # Verify control scenario had 0 alarms before and after window 15
        assert "| control          | DataDrift          | None                 | None              |                       0 |                        0 |" in content
        assert "| control          | UnseenCategory     | None                 | None              |                       0 |                        0 |" in content
        assert "| control          | IsolationOutlier   | None                 | None              |                       0 |                        0 |" in content
        assert "| control          | FraudScoreDrift    | None                 | None              |                       0 |                        0 |" in content
        assert "| control          | PerformanceMonitor | None                 | None              |                       0 |                        0 |" in content

    def test_onset_window_validation(self):
        from src.observatory.config import ObservatoryConfig, DriftInjectionConfig
        with pytest.raises(ValueError, match="onset_window"):
            ObservatoryConfig(drift_injection=DriftInjectionConfig(onset_window=5))
