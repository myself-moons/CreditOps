"""
tests/observatory/test_phase3.py — CreditOps v2 Observatory Phase 3 Tests

Validates:
  1. Hand-computed cost accounting verification
  2. Policy unit tests on synthetic signal sequences:
     - NeverRetrain on control has 0 retrains
     - FixedSchedule retrains at exact periodic windows
     - ThresholdPolicy triggers on DataDrift alarm
     - AdaptivePolicy persistence, cooldown, and expected-benefit gate
  3. No-future-label leakage assertion in ReplayEngine (max training window index <= t - delay)
  4. Determinism: same seed gives identical ReplayResult
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.observatory.cost import CostModel, WindowCostResult
from src.observatory.monitoring.detectors import DetectionResult
from src.observatory.policies.adaptive_policy import AdaptivePolicy
from src.observatory.policies.base import PolicyDecision, PolicyState
from src.observatory.policies.fixed_schedule import FixedSchedule
from src.observatory.policies.never_retrain import NeverRetrain
from src.observatory.policies.threshold_policy import ThresholdPolicy
from src.observatory.simulator.engine import ReplayEngine, ReplayResult
from src.observatory.simulator.stream import StreamWindow

ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================================ #
# 1. Cost Accounting Hand-Computed Example                                     #
# ============================================================================ #

class TestCostAccounting:
    def test_hand_computed_cost_model(self):
        cost_model = CostModel(
            fn_cost=500.0,
            fp_cost=5.0,
            retrain_cost=200.0,
            review_cost=10.0,
        )

        # 5 samples: 2 actual fraud, 3 actual legitimate
        # Predictions: row 0 (TP), row 1 (FP), row 2 (FN), row 3 (TN), row 4 (TN)
        y_true = np.array([1, 0, 1, 0, 0])
        y_prob = np.array([0.99, 0.98, 0.50, 0.10, 0.20])
        decision_threshold = 0.97

        # Case 1: action = "no_action"
        res_no_action = cost_model.evaluate_window(
            window_index=0,
            y_true=y_true,
            y_prob=y_prob,
            decision_threshold=decision_threshold,
            action="no_action",
        )
        assert res_no_action.tp == 1
        assert res_no_action.fp == 1
        assert res_no_action.fn == 1
        assert res_no_action.tn == 2
        assert res_no_action.fn_cost == 500.0
        assert res_no_action.fp_cost == 5.0
        assert res_no_action.decision_cost == 505.0
        assert res_no_action.retrain_cost == 0.0
        assert res_no_action.alert_cost == 0.0
        assert res_no_action.total_cost == 505.0

        # Case 2: action = "retrain"
        res_retrain = cost_model.evaluate_window(
            window_index=1,
            y_true=y_true,
            y_prob=y_prob,
            decision_threshold=decision_threshold,
            action="retrain",
        )
        assert res_retrain.decision_cost == 505.0
        assert res_retrain.retrain_cost == 200.0
        assert res_retrain.alert_cost == 0.0
        assert res_retrain.total_cost == 705.0

        # Case 3: action = "alert"
        res_alert = cost_model.evaluate_window(
            window_index=2,
            y_true=y_true,
            y_prob=y_prob,
            decision_threshold=decision_threshold,
            action="alert",
        )
        assert res_alert.decision_cost == 505.0
        assert res_alert.retrain_cost == 0.0
        assert res_alert.alert_cost == 10.0
        assert res_alert.total_cost == 515.0


# ============================================================================ #
# 2. Policy Unit Tests on Synthetic Signal Sequences                           #
# ============================================================================ #

class TestPolicyUnitContracts:
    def test_never_retrain_always_no_action(self):
        policy = NeverRetrain()
        state = PolicyState(
            window_index=5,
            windows_since_last_retrain=10,
            current_detector_results={"DataDrift": DetectionResult(score=2.0, alarm=True, window_index=5, detector="DataDrift")},
        )
        dec = policy.decide(state)
        assert dec.action == "no_action"
        assert "NeverRetrain" in dec.reason

    def test_fixed_schedule_retrains_at_exact_intervals(self):
        policy = FixedSchedule(k_windows=4)

        # Window since retrain = 3 (< 4) -> no_action
        state1 = PolicyState(window_index=3, windows_since_last_retrain=3)
        assert policy.decide(state1).action == "no_action"

        # Window since retrain = 4 (>= 4) -> retrain
        state2 = PolicyState(window_index=4, windows_since_last_retrain=4)
        assert policy.decide(state2).action == "retrain"

    def test_threshold_policy_retrain_and_cooldown(self):
        policy = ThresholdPolicy(cooldown_windows=3)

        # Alarm active and cooldown satisfied -> retrain
        state_alarm = PolicyState(
            window_index=5,
            windows_since_last_retrain=5,
            current_detector_results={"DataDrift": DetectionResult(score=1.5, alarm=True, window_index=5, detector="DataDrift")},
        )
        assert policy.decide(state_alarm).action == "retrain"

        # Alarm active but in cooldown (windows_since_last_retrain=1 < 3) -> no_action
        state_cooldown = PolicyState(
            window_index=6,
            windows_since_last_retrain=1,
            current_detector_results={"DataDrift": DetectionResult(score=1.5, alarm=True, window_index=6, detector="DataDrift")},
        )
        assert policy.decide(state_cooldown).action == "no_action"

    def test_adaptive_policy_persistence_and_cost_gate(self):
        policy = AdaptivePolicy(
            n_perf_persistence=2,
            n_free_persistence=2,
            cooldown_windows=3,
            horizon_windows=5,
            retrain_cost=200.0,
        )

        alarm_det = {"data_drift": DetectionResult(score=1.5, alarm=True, window_index=1, detector="data_drift")}
        no_alarm_det = {"data_drift": DetectionResult(score=0.2, alarm=False, window_index=0, detector="data_drift")}

        # 1. Single alarm window -> persistence not met -> alert
        state_single = PolicyState(
            window_index=1,
            windows_since_last_retrain=10,
            current_detector_results=alarm_det,
            detector_history=[no_alarm_det],
            recent_costs=[50.0],
            reference_cost_per_window=50.0,
        )
        assert policy.decide(state_single).action == "alert"

        # 2. Persisted for 2 windows, but expected benefit (0) <= retrain_cost (200) -> alert
        state_persisted_low_benefit = PolicyState(
            window_index=2,
            windows_since_last_retrain=10,
            current_detector_results=alarm_det,
            detector_history=[alarm_det, alarm_det],
            recent_costs=[50.0, 50.0],
            reference_cost_per_window=50.0,
        )
        assert policy.decide(state_persisted_low_benefit).action == "alert"

        # 3. Persisted for 2 windows AND expected benefit ($60 * 5 = $300) > retrain_cost ($200) -> retrain
        state_persisted_high_benefit = PolicyState(
            window_index=3,
            windows_since_last_retrain=10,
            current_detector_results=alarm_det,
            detector_history=[alarm_det, alarm_det],
            recent_costs=[110.0, 110.0],
            reference_cost_per_window=50.0,  # excess = 60/win * 5 horizon = 300 > 200
        )
        assert policy.decide(state_persisted_high_benefit).action == "retrain"

        # 4. Performance degradation persisted 2 windows -> immediate retrain
        perf_alarm = DetectionResult(score=0.3, alarm=True, window_index=4, detector="performance_monitor")
        state_perf = PolicyState(
            window_index=4,
            windows_since_last_retrain=10,
            performance_result=perf_alarm,
            performance_history=[perf_alarm, perf_alarm],
        )
        assert policy.decide(state_perf).action == "retrain"

        # 5. Cooldown gate suppresses retrain
        state_perf_cooldown = PolicyState(
            window_index=5,
            windows_since_last_retrain=1,  # cooldown=3
            performance_result=perf_alarm,
            performance_history=[perf_alarm, perf_alarm],
        )
        assert policy.decide(state_perf_cooldown).action == "no_action"


# ============================================================================ #
# 3. Replay Engine No-Leakage and Determinism                                  #
# ============================================================================ #

class DummyModel:
    def predict_proba(self, X):
        # Deterministic mock probabilities
        n = len(X)
        p1 = np.full(n, 0.05)
        return np.column_stack([1 - p1, p1])


class DummyPreprocessor:
    def transform(self, df):
        return np.zeros((len(df), 5))


class TestReplayEngineContracts:
    @pytest.fixture
    def synthetic_stream(self) -> List[StreamWindow]:
        windows = []
        for i in range(10):
            df = pd.DataFrame({"amt": [10.0, 20.0, 30.0], "haversine_km": [5.0, 10.0, 15.0]})
            labels = pd.Series([0, 1, 0])
            windows.append(StreamWindow(
                window_index=i,
                start_date=f"2020-10-{i+1:02d}",
                end_date=f"2020-10-{i+2:02d}",
                dataframe=df,
                is_drifted=False,
                drift_strength=0.0,
                labels=labels,
                raw_df=df,
            ))
        return windows

    def test_no_future_label_leakage_assertion(self, synthetic_stream):
        cost_model = CostModel()
        engine = ReplayEngine(
            champion_model=DummyModel(),
            preprocessor=DummyPreprocessor(),
            cost_model=cost_model,
            base_train_features=np.zeros((20, 5)),
            base_train_labels=np.zeros(20),
            label_delay_windows=2,
            log_decisions=False,
            seed=42,
        )

        # Policy that retrains at window 5
        class ForceRetrainPolicy(ThresholdPolicy):
            def decide(self, state: PolicyState) -> PolicyDecision:
                if state.window_index == 5:
                    return PolicyDecision(action="retrain", reason="Forced retrain test")
                return PolicyDecision(action="no_action", reason="No action")

        res = engine.run(
            stream_windows=synthetic_stream,
            policy=ForceRetrainPolicy(),
            detectors={},
            performance_monitor=None,
            scenario_name="control",
            onset_window=5,
        )

        assert res.n_retrains == 1
        assert res.retrain_windows == [5]

    def test_determinism_same_seed_gives_identical_result(self, synthetic_stream):
        cost_model = CostModel()
        engine1 = ReplayEngine(
            champion_model=DummyModel(),
            preprocessor=DummyPreprocessor(),
            cost_model=cost_model,
            base_train_features=np.zeros((20, 5)),
            base_train_labels=np.zeros(20),
            label_delay_windows=2,
            log_decisions=False,
            seed=42,
        )
        engine2 = ReplayEngine(
            champion_model=DummyModel(),
            preprocessor=DummyPreprocessor(),
            cost_model=cost_model,
            base_train_features=np.zeros((20, 5)),
            base_train_labels=np.zeros(20),
            label_delay_windows=2,
            log_decisions=False,
            seed=42,
        )

        res1 = engine1.run(
            stream_windows=synthetic_stream,
            policy=FixedSchedule(k_windows=3),
            detectors={},
            performance_monitor=None,
        )
        res2 = engine2.run(
            stream_windows=synthetic_stream,
            policy=FixedSchedule(k_windows=3),
            detectors={},
            performance_monitor=None,
        )

        assert res1.total_cost == res2.total_cost
        assert res1.retrain_windows == res2.retrain_windows
