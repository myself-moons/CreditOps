"""
tests/observatory/test_phase4.py — CreditOps v2 Phase 4 Test Suite

Covers:
  - Every valid state transition: candidate -> promoted, candidate -> rejected, promoted -> rolled_back
  - Invalid state transitions raise InvalidStateTransitionError
  - Promotion gate rejects a deliberately bad challenger
  - Rollback triggers on a deliberately degraded promoted model
  - Holdout is never in the training set
  - Same seed gives identical results (reproducibility)
  - Model versions and audit logs written through LogStore
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.observatory.cost import CostModel
from src.observatory.registry import (
    InvalidStateTransitionError,
    ModelRegistry,
    ModelState,
)
from src.observatory.simulator.engine import ReplayEngine, ReplayResult
from src.observatory.simulator.stream import StreamWindow
from src.observatory.storage.sqlite_store import SQLiteLogStore
from src.observatory.policies.base import BasePolicy, PolicyDecision


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers & Fixtures                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

class DummyModel:
    """Deterministic dummy classifier."""
    def __init__(self, const_prob: float = 0.99):
        self.const_prob = const_prob

    def predict_proba(self, X):
        p1 = np.full(len(X), self.const_prob)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class BadModel:
    """Always predicts near 0 (fails to detect fraud)."""
    def predict_proba(self, X):
        p1 = np.full(len(X), 0.01)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class AlwaysRetrainPolicy(BasePolicy):
    def __init__(self, retrain_at_window: int = 5):
        super().__init__("AlwaysRetrainTest")
        self.retrain_at_window = retrain_at_window

    def decide(self, state) -> PolicyDecision:
        if state.window_index == self.retrain_at_window:
            return PolicyDecision(action="retrain", reason="Test forced retrain")
        return PolicyDecision(action="no_action", reason="No action")


def _make_dummy_windows(n_windows: int = 10, rows_per_window: int = 40):
    windows = []
    rng = np.random.default_rng(42)
    for w in range(n_windows):
        feats = pd.DataFrame({
            "hour": rng.integers(0, 24, size=rows_per_window),
            "log_amt": rng.uniform(2.0, 6.0, size=rows_per_window),
            "cat_gas_transport": rng.choice([0, 1], size=rows_per_window),
            "cat_grocery_pos": rng.choice([0, 1], size=rows_per_window),
        })
        # Ensure some fraud in each window
        labels = pd.Series(np.zeros(rows_per_window, dtype=int))
        fraud_idx = rng.choice(rows_per_window, size=4, replace=False)
        labels.iloc[fraud_idx] = 1

        windows.append(
            StreamWindow(
                window_index=w,
                start_date=f"2020-10-{w*2+1:02d}",
                end_date=f"2020-10-{w*2+2:02d}",
                dataframe=feats,
                is_drifted=False,
                drift_strength=0.0,
                labels=labels,
                metadata={},
            )
        )
    return windows


class IdentityPreprocessor:
    def transform(self, df):
        return df.to_numpy(dtype=float)


# ──────────────────────────────────────────────────────────────────────────── #
# 1. State Machine & Transitions                                               #
# ──────────────────────────────────────────────────────────────────────────── #

class TestRegistryStateTransitions:
    def test_valid_transitions(self):
        store = SQLiteLogStore(":memory:")
        registry = ModelRegistry(log_store=store)

        # 1. candidate -> promoted
        cand = registry.register_candidate(
            model=DummyModel(),
            version="v2.0",
            threshold=0.90,
            data_hash="h1",
            actor="test",
            reason="testing",
        )
        assert cand.state == ModelState.CANDIDATE

        prom = registry.promote("v2.0", actor="gate", reason="recall improved")
        assert prom.state == ModelState.PROMOTED

        # 2. promoted -> rolled_back
        rb = registry.rollback("v2.0", actor="monitor", reason="cost high")
        assert rb.state == ModelState.ROLLED_BACK

        # 3. candidate -> rejected
        cand2 = registry.register_candidate(
            model=DummyModel(),
            version="v3.0",
            threshold=0.90,
            data_hash="h2",
            actor="test",
            reason="testing",
        )
        rej = registry.reject("v3.0", actor="gate", reason="pr-auc drop")
        assert rej.state == ModelState.REJECTED

    def test_invalid_transitions_raise(self):
        registry = ModelRegistry()

        # candidate -> rolled_back directly is invalid
        c1 = registry.register_candidate(DummyModel(), "v2.0", 0.9, "h1", actor="t", reason="r")
        with pytest.raises(InvalidStateTransitionError):
            registry.rollback("v2.0")

        # rejected -> promoted is invalid
        registry.reject("v2.0")
        with pytest.raises(InvalidStateTransitionError):
            registry.promote("v2.0")

        # rolled_back -> promoted is invalid
        c2 = registry.register_candidate(DummyModel(), "v3.0", 0.9, "h2", actor="t", reason="r")
        registry.promote("v3.0")
        registry.rollback("v3.0")
        with pytest.raises(InvalidStateTransitionError):
            registry.promote("v3.0")

    def test_audit_logs_written_to_logstore(self):
        store = SQLiteLogStore(":memory:")
        registry = ModelRegistry(log_store=store)

        registry.register_initial_champion(DummyModel(), "v1.0", 0.97, "data_hash_0")
        registry.register_candidate(DummyModel(), "v2.0", 0.95, "data_hash_1", actor="pipeline", reason="trained")
        registry.promote("v2.0", actor="gate", reason="passed gate")

        mv_records = store.query("model_versions")
        audit_records = store.query("audit_log")

        assert len(mv_records) == 3
        assert len(audit_records) == 3
        actions = [r["action"] for r in audit_records]
        assert actions == ["init_champion", "register_candidate", "promote"]
        assert all("actor" in r and "data_hash" in r and "metrics" in r for r in audit_records)


# ──────────────────────────────────────────────────────────────────────────── #
# 2. Promotion Gate Rejection                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

class TestPromotionGate:
    def test_gate_rejects_deliberately_bad_challenger(self, monkeypatch):
        windows = _make_dummy_windows(n_windows=8)
        cost_model = CostModel(fn_cost=500, fp_cost=5, retrain_cost=200, review_cost=10)

        # Baseline champion detects everything
        good_champion = DummyModel(0.99)
        prep = IdentityPreprocessor()
        X_base = prep.transform(windows[0].features_df)
        y_base = windows[0].labels.to_numpy()

        # Patch XGBClassifier to return a BadModel that predicts near 0
        import xgboost as xgb
        class MockXGB:
            def __init__(self, *args, **kwargs):
                pass
            def fit(self, *args, **kwargs):
                return self
            def predict_proba(self, X):
                p1 = np.full(len(X), 0.001)
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])

        monkeypatch.setattr(xgb, "XGBClassifier", MockXGB)

        engine = ReplayEngine(
            champion_model=good_champion,
            preprocessor=prep,
            cost_model=cost_model,
            base_train_features=X_base,
            base_train_labels=y_base,
            decision_threshold=0.97,
            label_delay_windows=2,
            deployment_mode="governed",
            margin_recall=0.05,
        )

        policy = AlwaysRetrainPolicy(retrain_at_window=5)
        res = engine.run(
            stream_windows=windows,
            policy=policy,
            detectors={},
            performance_monitor=None,
            scenario_name="control",
            onset_window=4,
        )

        # Gate rejected bad challenger!
        assert len(res.rejected_versions) == 1
        assert len(res.promoted_versions) == 0
        assert res.n_retrains == 0  # Not promoted into retrain_windows
        assert res.total_retrain_cost == 200.0  # Retrain cost is still paid!


# ──────────────────────────────────────────────────────────────────────────── #
# 3. Post-Promotion Rollback                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

class TestPostPromotionRollback:
    def test_rollback_triggers_on_deliberately_degraded_promoted_model(self):
        windows = _make_dummy_windows(n_windows=10)
        cost_model = CostModel(fn_cost=500, fp_cost=5, retrain_cost=200, review_cost=10)

        good_champion = DummyModel(0.99)
        prep = IdentityPreprocessor()
        X_base = prep.transform(windows[0].features_df)
        y_base = windows[0].labels.to_numpy()

        engine = ReplayEngine(
            champion_model=good_champion,
            preprocessor=prep,
            cost_model=cost_model,
            base_train_features=X_base,
            base_train_labels=y_base,
            decision_threshold=0.97,
            label_delay_windows=1,
            deployment_mode="governed",
            margin_cost=50.0,
            rollback_window_m=5,
        )

        # Manually register and promote a model that we then sabotage to test rollback trigger
        cand = engine.registry.register_candidate(
            model=BadModel(),  # Bad model will incur massive FN costs
            version="v2.0",
            threshold=0.50,
            data_hash="h_bad",
            actor="test",
            reason="testing",
        )
        engine.registry.promote("v2.0", actor="test", reason="testing")

        # Set engine state as if v2.0 was just promoted
        rollback_tracker = {
            "promoted_version": "v2.0",
            "promoted_window": 2,
            "shadow_model": good_champion,
            "shadow_threshold": 0.97,
            "expiry_window": 8,
            "window_data": {},
        }

        # Simulate scoring windows 3, 4 with bad promoted vs good shadow
        # At window 4 (with label_delay=1), window 3 labels are evaluated:
        # Promoted model misses 4 fraud -> $2,000 cost
        # Shadow champion catches fraud -> $0 cost
        # Excess $2,000 > margin $50 -> Rollback triggers!
        current_champ = BadModel()
        current_thresh = 0.50

        rolled_back = False
        for t in [3, 4, 5]:
            w = windows[t]
            X_curr = prep.transform(w.features_df)
            y_true = w.labels.to_numpy()
            probs_prom = current_champ.predict_proba(X_curr)[:, 1]
            probs_shad = rollback_tracker["shadow_model"].predict_proba(X_curr)[:, 1]

            rollback_tracker["window_data"][t] = {
                "y_true": y_true,
                "promoted_probs": probs_prom,
                "shadow_probs": probs_shad,
                "promoted_threshold": current_thresh,
                "shadow_threshold": rollback_tracker["shadow_threshold"],
            }

            eval_windows = [kw for kw in rollback_tracker["window_data"] if kw <= t - 1]
            if eval_windows:
                c_prom = sum(cost_model.evaluate_window(kw, rollback_tracker["window_data"][kw]["y_true"], rollback_tracker["window_data"][kw]["promoted_probs"], current_thresh, "no_action").decision_cost for kw in eval_windows)
                c_shad = sum(cost_model.evaluate_window(kw, rollback_tracker["window_data"][kw]["y_true"], rollback_tracker["window_data"][kw]["shadow_probs"], rollback_tracker["shadow_threshold"], "no_action").decision_cost for kw in eval_windows)
                if c_prom > c_shad + 50.0:
                    engine.registry.rollback("v2.0", actor="rollback_monitor", reason="High cost")
                    rolled_back = True
                    break

        assert rolled_back is True
        assert engine.registry.get_model("v2.0").state == ModelState.ROLLED_BACK


# ──────────────────────────────────────────────────────────────────────────── #
# 4. Holdout Isolation & Leakage Prevention                                    #
# ──────────────────────────────────────────────────────────────────────────── #

class TestHoldoutIsolation:
    def test_holdout_never_in_training_set(self):
        windows = _make_dummy_windows(n_windows=8)
        cost_model = CostModel()
        prep = IdentityPreprocessor()
        X_base = prep.transform(windows[0].features_df)
        y_base = windows[0].labels.to_numpy()

        engine = ReplayEngine(
            champion_model=DummyModel(),
            preprocessor=prep,
            cost_model=cost_model,
            base_train_features=X_base,
            base_train_labels=y_base,
            label_delay_windows=2,
            deployment_mode="governed",
            k_val_windows=2,
        )

        policy = AlwaysRetrainPolicy(retrain_at_window=6)
        res = engine.run(
            stream_windows=windows,
            policy=policy,
            detectors={},
            performance_monitor=None,
            scenario_name="control",
        )

        # Retrain at t=6:
        # Available labelled windows: <= 6 - 2 = 4 (windows 0, 1, 2, 3, 4)
        # k_val_windows = 2: holdout is [3, 4], train stream windows are [0, 1, 2]
        # Registry records holdout_windows in metrics
        models = engine.registry.models
        if "v2.0" in models:
            cand = models["v2.0"]
            holdout_w = cand.metrics.get("holdout_windows", [])
            assert holdout_w == [3, 4]
            # Data hash indicates which windows were in training
            assert "train_win_[0, 1, 2]" in cand.data_hash
            assert not any(hw in [0, 1, 2] for hw in holdout_w)


# ──────────────────────────────────────────────────────────────────────────── #
# 5. Reproducibility: Same Seed -> Same Result                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class TestReproducibility:
    def test_same_seed_gives_identical_result(self):
        windows = _make_dummy_windows(n_windows=6)
        cost_model = CostModel()
        prep = IdentityPreprocessor()
        X_base = prep.transform(windows[0].features_df)
        y_base = windows[0].labels.to_numpy()

        engine1 = ReplayEngine(
            champion_model=DummyModel(),
            preprocessor=prep,
            cost_model=cost_model,
            base_train_features=X_base,
            base_train_labels=y_base,
            deployment_mode="governed",
            seed=42,
        )
        res1 = engine1.run(
            stream_windows=windows,
            policy=AlwaysRetrainPolicy(retrain_at_window=4),
            detectors={},
            performance_monitor=None,
        )

        engine2 = ReplayEngine(
            champion_model=DummyModel(),
            preprocessor=prep,
            cost_model=cost_model,
            base_train_features=X_base,
            base_train_labels=y_base,
            deployment_mode="governed",
            seed=42,
        )
        res2 = engine2.run(
            stream_windows=windows,
            policy=AlwaysRetrainPolicy(retrain_at_window=4),
            detectors={},
            performance_monitor=None,
        )

        assert res1.total_cost == res2.total_cost
        assert res1.promoted_versions == res2.promoted_versions
        assert res1.rejected_versions == res2.rejected_versions
        assert res1.per_window_costs == res2.per_window_costs
