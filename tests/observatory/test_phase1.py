"""
tests/observatory/test_phase1.py — CreditOps v2 Observatory, Phase 1

Comprehensive test suite verifying Phase 1 requirements:
  1. Same seed gives identical output; different seed gives different output
  2. Control scenario leaves data completely unchanged
  3. Windows before onset are unchanged; windows after are changed
  4. is_drifted and onset_window ground truth are correct (sudden & gradual)
  5. fraud_rate_shift reaches target rate within tolerance via oversampling real fraud
  6. novel_pattern produces an unseen category and preprocessing handles it without crashing
  7. Empty window and missing-column inputs fail gracefully with clear error
  8. StreamWindow dict & object access, warning when fraud < N, and fallback on few windows
  9. Scenario manifest reproducibility (same seed + config produces identical manifest)
 10. V1 isolation: Observatory does not import V1 app / model training
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.observatory.config import (
    DriftInjectionConfig,
    ObservatoryConfig,
    load_config,
)
from src.observatory.drift.injector import (
    BASE_FRAUD_RATE,
    _effective_magnitude,
    apply_drift,
)
from src.observatory.simulator.stream import (
    StreamWindow,
    WindowedStream,
)
from src.observatory.simulator.manifest import (
    write_scenario_manifest,
    compute_slice_hash,
)
from src.features import build_features, feature_names
from src.data_preprocessing import build_preprocessor


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

def _make_raw_window(n: int = 500, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic raw transactions suitable for build_features()."""
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n * fraud_rate))
    labels = [1] * n_fraud + [0] * (n - n_fraud)
    rng.shuffle(labels)

    categories = [
        "misc_net", "grocery_pos", "entertainment", "gas_transport",
        "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
        "food_dining", "personal_care", "health_fitness", "travel",
        "kids_pets", "home",
    ]

    return pd.DataFrame({
        "trans_date_trans_time": pd.date_range("2020-10-01", periods=n, freq="10min"),
        "amt": rng.uniform(5.0, 300.0, n),
        "category": rng.choice(categories, n),
        "gender": rng.choice(["M", "F"], n),
        "dob": ["1985-06-15"] * n,
        "lat": rng.uniform(30.0, 45.0, n),
        "long": rng.uniform(-100.0, -75.0, n),
        "city_pop": rng.integers(1000, 500000, n).astype(float),
        "merch_lat": rng.uniform(30.0, 45.0, n),
        "merch_long": rng.uniform(-100.0, -75.0, n),
        "is_fraud": labels,
    })


def _build(df: pd.DataFrame):
    feats = build_features(df)
    labels = df["is_fraud"].reset_index(drop=True)
    return feats, labels


# ──────────────────────────────────────────────────────────────────────────── #
# 1. Config tests                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

class TestObservatoryConfig:
    def test_defaults(self):
        cfg = ObservatoryConfig()
        assert cfg.seed == 42
        assert cfg.window_size_days == 2
        assert cfg.min_fraud_per_window == 5
        assert cfg.min_windows == 10
        assert cfg.drift_injection.scenario == "control"
        assert cfg.drift_injection.shape == "sudden"

    def test_load_from_yaml(self, tmp_path):
        yaml_content = {
            "observatory": {
                "seed": 99,
                "window_size_days": 3,
                "stream_start": "2020-09-01",
                "stream_end": "2020-12-31",
                "min_fraud_per_window": 8,
                "min_windows": 12,
                "drift_injection": {
                    "scenario": "fraud_rate_shift",
                    "shape": "gradual",
                    "onset_window": 10,
                    "ramp_windows": 4,
                    "magnitude": 2.5,
                    "affected_categories": ["travel"],
                },
            },
            "cost_model": {"fn_cost": 300, "fp_cost": 3, "retrain_cost": 100, "review_cost": 5},
        }
        p = tmp_path / "observatory.yaml"
        p.write_text(yaml.dump(yaml_content))
        cfg = load_config(p)
        assert cfg.seed == 99
        assert cfg.window_size_days == 3
        assert cfg.min_fraud_per_window == 8
        assert cfg.min_windows == 12
        assert cfg.drift_injection.scenario == "fraud_rate_shift"
        assert cfg.drift_injection.shape == "gradual"
        assert cfg.drift_injection.onset_window == 10
        assert cfg.drift_injection.magnitude == pytest.approx(2.5)

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            DriftInjectionConfig(scenario="nonexistent")

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            DriftInjectionConfig(shape="wavy")


# ──────────────────────────────────────────────────────────────────────────── #
# 2. Reproducibility: Same seed / Different seed                               #
# ──────────────────────────────────────────────────────────────────── #

class TestReproducibility:
    def test_same_seed_gives_identical_output(self):
        raw = _make_raw_window(n=200, fraud_rate=0.04, seed=1)
        feats, labels = _build(raw)

        for scenario in ["covariate_shift", "fraud_rate_shift", "concept_drift", "novel_pattern"]:
            f1, l1 = apply_drift(
                feats, labels, window_index=5,
                scenario=scenario, shape="sudden", onset_window=0,
                ramp_windows=1, magnitude=2.0, seed=42,
            )
            f2, l2 = apply_drift(
                feats, labels, window_index=5,
                scenario=scenario, shape="sudden", onset_window=0,
                ramp_windows=1, magnitude=2.0, seed=42,
            )
            pd.testing.assert_frame_equal(f1, f2)
            pd.testing.assert_series_equal(l1, l2)

    def test_different_seed_gives_different_output(self):
        raw = _make_raw_window(n=200, fraud_rate=0.04, seed=1)
        feats, labels = _build(raw)

        f1, _ = apply_drift(
            feats, labels, window_index=5,
            scenario="covariate_shift", shape="sudden", onset_window=0,
            ramp_windows=1, magnitude=2.0, seed=42,
        )
        f2, _ = apply_drift(
            feats, labels, window_index=5,
            scenario="covariate_shift", shape="sudden", onset_window=0,
            ramp_windows=1, magnitude=2.0, seed=99,
        )
        assert not f1["log_amt"].equals(f2["log_amt"])


# ──────────────────────────────────────────────────────────────────────────── #
# 3. Control & Onset boundaries                                               #
# ──────────────────────────────────────────────────────────────────────────── #

class TestControlAndOnset:
    @pytest.fixture
    def window(self):
        raw = _make_raw_window(n=200, fraud_rate=0.04, seed=10)
        return _build(raw)

    def test_control_scenario_leaves_data_unchanged(self, window):
        feats, labels = window
        f_out, l_out = apply_drift(
            feats, labels, window_index=15,
            scenario="control", shape="sudden", onset_window=5,
            ramp_windows=1, magnitude=3.0, seed=42,
        )
        pd.testing.assert_frame_equal(feats, f_out)
        pd.testing.assert_series_equal(labels, l_out)

    def test_windows_before_onset_are_unchanged(self, window):
        feats, labels = window
        for scenario in ["covariate_shift", "fraud_rate_shift", "concept_drift", "novel_pattern"]:
            f_out, l_out = apply_drift(
                feats, labels, window_index=4,
                scenario=scenario, shape="sudden", onset_window=5,
                ramp_windows=1, magnitude=2.0, seed=42,
            )
            pd.testing.assert_frame_equal(feats, f_out)
            pd.testing.assert_series_equal(labels, l_out)

    def test_windows_after_onset_are_changed(self, window):
        feats, labels = window
        f_out, l_out = apply_drift(
            feats, labels, window_index=5,
            scenario="covariate_shift", shape="sudden", onset_window=5,
            ramp_windows=1, magnitude=2.0, seed=42,
        )
        assert not f_out.equals(feats)


# ──────────────────────────────────────────────────────────────────────────── #
# 4. is_drifted and onset_window ground truth (sudden & gradual)               #
# ──────────────────────────────────────────────────────────────────────────── #

class TestGroundTruthAndGradual:
    def test_effective_magnitude_sudden(self):
        assert _effective_magnitude(4, onset_window=5, ramp_windows=3, magnitude=2.0, shape="sudden") == 0.0
        assert _effective_magnitude(5, onset_window=5, ramp_windows=3, magnitude=2.0, shape="sudden") == 2.0
        assert _effective_magnitude(10, onset_window=5, ramp_windows=3, magnitude=2.0, shape="sudden") == 2.0

    def test_effective_magnitude_gradual(self):
        assert _effective_magnitude(4, onset_window=5, ramp_windows=5, magnitude=4.0, shape="gradual") == 0.0
        # at onset (step 0/4): 0.0
        assert _effective_magnitude(5, onset_window=5, ramp_windows=5, magnitude=4.0, shape="gradual") == 0.0
        # mid ramp (step 2/4): 2.0
        assert _effective_magnitude(7, onset_window=5, ramp_windows=5, magnitude=4.0, shape="gradual") == pytest.approx(2.0)
        # full ramp (step 4/4): 4.0
        assert _effective_magnitude(9, onset_window=5, ramp_windows=5, magnitude=4.0, shape="gradual") == pytest.approx(4.0)
        # after ramp: 4.0
        assert _effective_magnitude(12, onset_window=5, ramp_windows=5, magnitude=4.0, shape="gradual") == pytest.approx(4.0)


# ──────────────────────────────────────────────────────────────────────────── #
# 5. fraud_rate_shift reaches target rate within tolerance                      #
# ──────────────────────────────────────────────────────────────────────────── #

class TestFraudRateShift:
    def test_fraud_rate_shift_reaches_target_rate_tolerance(self):
        # Window with baseline Sparkov fraud rate (~0.58%)
        # 1000 txns, 6 fraud
        raw = _make_raw_window(n=1000, fraud_rate=0.006, seed=20)
        feats, labels = _build(raw)

        # Scale by magnitude = 3.0 (target: 3 * 0.005789 = 0.017367 or ~1.74%)
        target_rate = 3.0 * BASE_FRAUD_RATE

        f_out, l_out = apply_drift(
            feats, labels, window_index=5,
            scenario="fraud_rate_shift", shape="sudden", onset_window=0,
            ramp_windows=1, magnitude=3.0, seed=42,
        )

        actual_rate = l_out.mean()
        assert abs(actual_rate - target_rate) < 0.005, f"Expected {target_rate:.4f}, got {actual_rate:.4f}"

    def test_fraud_rate_shift_does_not_fabricate_labels_on_legit_rows(self):
        raw = _make_raw_window(n=500, fraud_rate=0.01, seed=21)
        feats, labels = _build(raw)

        legit_original_count = (labels == 0).sum()
        _, l_out = apply_drift(
            feats, labels, window_index=5,
            scenario="fraud_rate_shift", shape="sudden", onset_window=0,
            ramp_windows=1, magnitude=3.0, seed=42,
        )

        # Oversampling real fraud rows should preserve legitimate rows intact
        legit_post_count = (l_out == 0).sum()
        assert legit_post_count == legit_original_count, "Legitimate rows must never be relabeled as fraud!"

    def test_fraud_rate_shift_post_onset_exceeds_control_by_target_factor(self):
        raw = _make_raw_window(n=1000, fraud_rate=0.005, seed=22)
        feats, labels = _build(raw)

        # Control leaves rate unchanged
        _, l_ctrl = apply_drift(
            feats, labels, window_index=5,
            scenario="control", shape="sudden", onset_window=0,
            ramp_windows=1, magnitude=3.0, seed=42,
        )
        ctrl_rate = l_ctrl.mean()

        # Fraud rate shift with mag=3.0
        _, l_drift = apply_drift(
            feats, labels, window_index=5,
            scenario="fraud_rate_shift", shape="sudden", onset_window=0,
            ramp_windows=1, magnitude=3.0, seed=42,
        )
        drift_rate = l_drift.mean()

        # Ratio must be at least target factor (3.0) within tolerance
        ratio = drift_rate / ctrl_rate
        assert ratio >= 2.8, f"Expected ratio >= 2.8, got {ratio:.2f}"


# ──────────────────────────────────────────────────────────────────────────── #
# 6. novel_pattern unseen category & preprocessing                             #
# ──────────────────────────────────────────────────────────────────────────── #

class TestNovelPattern:
    def test_novel_pattern_unseen_category_and_preprocessing(self):
        raw = _make_raw_window(n=200, fraud_rate=0.05, seed=30)
        feats, labels = _build(raw)

        f_out, l_out = apply_drift(
            feats, labels, window_index=5,
            scenario="novel_pattern", shape="sudden", onset_window=0,
            ramp_windows=1, magnitude=1.0, seed=42,
        )

        fraud_indices = l_out[l_out == 1].index
        assert len(fraud_indices) > 0

        # Unseen merchant category: all 14 cat_* columns are 0 for affected fraud rows
        cat_cols = [c for c in f_out.columns if c.startswith("cat_")]
        first_fraud_cats = f_out.loc[fraud_indices[0], cat_cols]
        assert (first_fraud_cats == 0).all(), "Unseen merchant category must zero out all known category one-hots"

        # Fit preprocessor on original training-like features, then transform novel pattern features
        preproc = build_preprocessor()
        preproc.fit(feats)
        # Preprocessor must transform without crashing on the unseen category representation
        X_enc = preproc.transform(f_out)
        assert X_enc.shape[0] == len(f_out)
        assert not np.isnan(X_enc).any()

    def test_raw_unseen_category_in_build_features_does_not_crash(self):
        raw = _make_raw_window(n=20, fraud_rate=0.1, seed=31)
        # Introduce novel category unseen in training
        raw.loc[0, "category"] = "unseen_crypto_gateway"
        feats = build_features(raw)

        cat_cols = [c for c in feats.columns if c.startswith("cat_")]
        assert len(cat_cols) == 14
        assert (feats.loc[0, cat_cols] == 0).all()

        preproc = build_preprocessor()
        preproc.fit(feats)
        X_enc = preproc.transform(feats)
        assert X_enc.shape[0] == len(feats)


# ──────────────────────────────────────────────────────────────────────────── #
# 7. Empty window and missing-column error handling                             #
# ──────────────────────────────────────────────────────────────────────────── #

class TestErrorHandling:
    def test_empty_window_fails_gracefully(self):
        empty_feats = pd.DataFrame(columns=feature_names())
        empty_labels = pd.Series([], dtype=int)

        f_out, l_out = apply_drift(
            empty_feats, empty_labels, window_index=5,
            scenario="covariate_shift", shape="sudden", onset_window=0,
            ramp_windows=1, magnitude=2.0, seed=42,
        )
        assert f_out.empty
        assert l_out.empty

    def test_missing_column_raises_clear_error(self):
        raw = _make_raw_window(n=20, fraud_rate=0.1, seed=32)
        feats, labels = _build(raw)
        # Drop a required column
        incomplete_feats = feats.drop(columns=["log_amt"])

        with pytest.raises(ValueError, match="missing required column: 'log_amt'"):
            apply_drift(
                incomplete_feats, labels, window_index=5,
                scenario="covariate_shift", shape="sudden", onset_window=0,
                ramp_windows=1, magnitude=2.0, seed=42,
            )


# ──────────────────────────────────────────────────────────────────────────── #
# 8. StreamWindow dict/object access, warning and fallback                      #
# ──────────────────────────────────────────────────────────────────────────── #

class TestStreamWindowAndFallback:
    def test_stream_window_object_and_dict_access(self):
        raw = _make_raw_window(n=50, fraud_rate=0.1, seed=40)
        cfg = ObservatoryConfig(
            seed=42, window_size_days=2,
            stream_start="2020-10-01", stream_end="2020-10-05",
            min_windows=1,
        )
        stream = WindowedStream(raw, cfg)
        windows = list(stream)
        assert len(windows) > 0

        w = windows[0]
        # Object attribute access
        assert hasattr(w, "window_index")
        assert hasattr(w, "start_date")
        assert hasattr(w, "end_date")
        assert hasattr(w, "dataframe")
        assert hasattr(w, "is_drifted")
        assert hasattr(w, "drift_strength")

        # Dict access
        assert w["window_index"] == w.window_index
        assert w["is_drifted"] == w.is_drifted
        assert isinstance(w["dataframe"], pd.DataFrame)
        assert isinstance(dict(w), dict)

    def test_low_fraud_warning(self):
        raw = _make_raw_window(n=50, fraud_rate=0.02, seed=41)
        raw["is_fraud"] = 0
        raw.loc[0, "is_fraud"] = 1  # only 1 fraud row < N=5
        cfg = ObservatoryConfig(
            seed=42, window_size_days=2,
            stream_start="2020-10-01", stream_end="2020-10-03",
            min_fraud_per_window=5,
            min_windows=1,
        )
        stream = WindowedStream(raw, cfg)
        with pytest.warns(UserWarning, match="fewer than threshold N=5"):
            list(stream)

    def test_fallback_shrinks_window_size_days(self):
        # 4 days of data: with window_size_days=2 yields 2-3 windows (< min_windows=6)
        raw = _make_raw_window(n=100, fraud_rate=0.05, seed=42)
        raw["trans_date_trans_time"] = pd.date_range("2020-10-01", "2020-10-04 23:59:59", periods=100)
        cfg = ObservatoryConfig(
            seed=42, window_size_days=2,
            stream_start="2020-10-01", stream_end="2020-10-04",
            min_windows=6,
        )
        stream = WindowedStream(raw, cfg)
        with pytest.warns(UserWarning, match="Fallback triggered: shrinking window_size_days"):
            windows = list(stream)
        assert len(windows) >= 3

    def test_stream_uses_real_data(self):
        # Slices real Sparkov dataset via DatasetAdapter
        cfg = ObservatoryConfig(
            seed=42, window_size_days=2,
            stream_start="2020-10-04", stream_end="2020-10-10",
            min_windows=1,
        )
        stream = WindowedStream(config=cfg)
        windows = list(stream)
        assert len(windows) > 0
        for w in windows:
            # Real Sparkov 2-day windows contain thousands of transactions
            assert w.n_rows >= 1000, f"Window {w.window_index} has only {w.n_rows} rows; expected real data >= 1000"


# ──────────────────────────────────────────────────────────────────────────── #
# 9. Scenario manifest reproducibility                                         #
# ──────────────────────────────────────────────────────────────────────────── #

class TestScenarioManifest:
    def test_manifest_reproducibility(self, tmp_path):
        raw = _make_raw_window(n=100, fraud_rate=0.05, seed=50)
        cfg = ObservatoryConfig(seed=42)

        p1 = write_scenario_manifest(cfg, n_windows=10, source_df=raw, out_dir=tmp_path / "run1")
        p2 = write_scenario_manifest(cfg, n_windows=10, source_df=raw, out_dir=tmp_path / "run2")

        m1 = json.loads(p1.read_text(encoding="utf-8"))
        m2 = json.loads(p2.read_text(encoding="utf-8"))

        assert m1["data_hash"] == m2["data_hash"]
        assert m1["scenario"] == m2["scenario"]
        assert m1["seed"] == m2["seed"]
        assert m1["n_windows"] == m2["n_windows"]


# ──────────────────────────────────────────────────────────────────────────── #
# 10. V1 isolation                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

class TestV1Isolation:
    def test_observatory_does_not_import_main_or_training(self):
        import importlib
        import src.observatory.config as _c
        import src.observatory.drift.injector as _i
        import src.observatory.simulator.stream as _s

        importlib.reload(_c)
        importlib.reload(_i)
        importlib.reload(_s)

        assert "src.main" not in sys.modules
        assert "src.model_training" not in sys.modules
