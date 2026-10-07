"""
src/observatory/simulator/engine.py — CreditOps v2 Replay Engine

Executes offline streaming simulation window-by-window:
  1. Score window t with current champion model (or shadow scoring)
  2. Compute decision metrics (FN, FP, TP, TN, costs, PR-AUC, recall)
  3. Execute drift detectors and delayed performance monitor
  4. Query policy for decision (no_action, alert, retrain)
  5. If retrain:
     - In naive mode: retrain on all available stream windows <= t - delay
     - In governed mode:
       * Hold out most recent K_val labelled windows (never trained on)
       * Train challenger on earlier labelled windows + base sample
       * Recalibrate challenger threshold to match FP budget
       * Promotion gate: challenger recall beats champion by margin_recall AND PR-AUC not lower
       * If promoted: activate challenger, keep champion in shadow for M windows with automatic rollback
       * If rejected: champion continues; retrain cost is paid
  6. Record decisions, model versions, and audit logs to LogStore
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve

from src.observatory.cost import CostModel, WindowCostResult
from src.observatory.monitoring.detectors import BaseDetector, DetectionResult
from src.observatory.monitoring.performance import PerformanceMonitor
from src.observatory.policies.base import BasePolicy, PolicyDecision, PolicyState
from src.observatory.registry import ModelRegistry, ModelState
from src.observatory.simulator.stream import StreamWindow
from src.observatory.storage.base import LogStore

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Summary metrics of a complete policy simulation run."""
    scenario: str
    policy: str
    seed: int
    onset_window: int
    n_retrains: int
    retrain_windows: List[int]
    detection_delay: Optional[int]
    false_alarms: int
    total_fn_fp_cost: float
    total_retrain_cost: float
    total_alert_cost: float
    total_cost: float
    loss_avoided_vs_never: float = 0.0
    net_benefit: float = 0.0
    mean_pr_auc_after_onset: float = 0.0
    duration_seconds: float = 0.0
    promoted_versions: List[str] = field(default_factory=list)
    rejected_versions: List[str] = field(default_factory=list)
    rolled_back_versions: List[str] = field(default_factory=list)
    per_window_costs: List[float] = field(default_factory=list, repr=False)
    per_window_pr_auc: List[float] = field(default_factory=list, repr=False)
    per_window_recall: List[float] = field(default_factory=list, repr=False)
    decisions: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReplayEngine:
    """Fast offline simulation engine with strict label delay and governed deployment."""

    def __init__(
        self,
        champion_model: Any,
        preprocessor: Any,
        cost_model: CostModel,
        base_train_features: np.ndarray,
        base_train_labels: np.ndarray,
        decision_threshold: float = 0.97,
        label_delay_windows: int = 2,
        stream_weight: float = 20.0,
        deployment_mode: str = "naive",
        log_store: Optional[LogStore] = None,
        log_decisions: bool = True,
        seed: int = 42,
        k_val_windows: int = 3,
        margin_recall: float = 0.02,
        margin_cost: float = 100.0,
        rollback_window_m: int = 5,
        fp_budget: Optional[float] = None,
    ) -> None:
        self.initial_champion = champion_model
        self.preprocessor = preprocessor
        self.cost_model = cost_model
        self.X_train_base = base_train_features
        self.y_train_base = base_train_labels
        self.decision_threshold = float(decision_threshold)
        self.label_delay = max(0, int(label_delay_windows))
        self.stream_weight = float(stream_weight)
        self.deployment_mode = str(deployment_mode)
        self.log_store = log_store
        self.log_decisions = bool(log_decisions)
        self.seed = int(seed)

        # Governed deployment settings
        self.k_val_windows = max(1, int(k_val_windows))
        self.margin_recall = float(margin_recall)
        self.margin_cost = float(margin_cost)
        self.rollback_window_m = max(1, int(rollback_window_m))
        self.fp_budget = float(fp_budget) if fp_budget is not None else None

        self.registry = ModelRegistry(log_store=log_store)

    def _compute_fp_budget(self, stream_windows: List[StreamWindow], ref_windows: int = 10) -> float:
        """Derive FP budget from champion's FPR at V1 threshold on reference windows 0-9."""
        if self.fp_budget is not None:
            return self.fp_budget

        ref_slice = stream_windows[:ref_windows]
        if not ref_slice:
            return 0.001

        fp_count = 0
        legit_count = 0
        for w in ref_slice:
            y_arr = w.labels.to_numpy()
            legit_mask = (y_arr == 0)
            legit_count += int(legit_mask.sum())
            if legit_mask.sum() > 0:
                X_w = self.preprocessor.transform(w.features_df.iloc[legit_mask])
                probs = self.initial_champion.predict_proba(X_w)[:, 1]
                fp_count += int((probs >= self.decision_threshold).sum())

        fpr = float(fp_count / max(1, legit_count))
        # Ensure budget is bounded reasonably
        return max(0.0001, min(0.10, fpr)) if fpr > 0 else 0.001

    def run(
        self,
        stream_windows: List[StreamWindow],
        policy: BasePolicy,
        detectors: Dict[str, BaseDetector],
        performance_monitor: PerformanceMonitor,
        scenario_name: str = "control",
        onset_window: int = 15,
        reference_cost_per_window: float = 0.0,
    ) -> ReplayResult:
        start_time = time.time()
        current_champion = self.initial_champion
        current_decision_threshold = self.decision_threshold

        # Register initial baseline champion in registry
        self.registry.register_initial_champion(
            model=self.initial_champion,
            version="v1.0",
            threshold=self.decision_threshold,
            data_hash="base_v1_data",
            metrics={"decision_threshold": self.decision_threshold},
        )

        fp_budget = self._compute_fp_budget(stream_windows)

        windows_since_last_retrain = 999
        retrain_windows: List[int] = []
        promoted_versions: List[str] = []
        rejected_versions: List[str] = []
        rolled_back_versions: List[str] = []
        model_version_counter = 1

        detector_history: List[Dict[str, DetectionResult]] = []
        performance_history: List[DetectionResult] = []
        recent_costs: List[float] = []

        window_costs: List[float] = []
        window_pr_aucs: List[float] = []
        window_recalls: List[float] = []
        recorded_decisions: List[Dict[str, Any]] = []

        # Shadow rollback tracker:
        # {
        #   "promoted_version": str,
        #   "promoted_window": int,
        #   "shadow_model": Any,
        #   "shadow_threshold": float,
        #   "expiry_window": int,
        #   "window_data": {w_idx: {"y_true": ..., "prom_probs": ..., "shad_probs": ...}}
        # }
        rollback_tracker: Optional[Dict[str, Any]] = None

        total_fn_fp = 0.0
        total_retrain = 0.0
        total_alert = 0.0

        for t, window in enumerate(stream_windows):
            X_curr = self.preprocessor.transform(window.features_df)
            y_true = window.labels.to_numpy()
            n_fraud = int(y_true.sum())

            # 1. Champion inference on current window
            probs = current_champion.predict_proba(X_curr)[:, 1]

            # 1b. Shadow scoring and rollback evaluation if active
            if rollback_tracker is not None:
                shad_model = rollback_tracker["shadow_model"]
                probs_shad = shad_model.predict_proba(X_curr)[:, 1]

                rollback_tracker["window_data"][t] = {
                    "y_true": y_true,
                    "promoted_probs": probs,
                    "shadow_probs": probs_shad,
                    "promoted_threshold": current_decision_threshold,
                    "shadow_threshold": rollback_tracker["shadow_threshold"],
                }

                # Evaluate cost on post-promotion windows whose delayed labels are now available
                eval_windows = [
                    w_idx for w_idx in rollback_tracker["window_data"]
                    if w_idx <= t - self.label_delay
                ]

                if eval_windows:
                    c_prom_tot = 0.0
                    c_shad_tot = 0.0
                    for kw in eval_windows:
                        wd = rollback_tracker["window_data"][kw]
                        c_p = self.cost_model.evaluate_window(
                            kw, wd["y_true"], wd["promoted_probs"], wd["promoted_threshold"], action="no_action"
                        ).decision_cost
                        c_s = self.cost_model.evaluate_window(
                            kw, wd["y_true"], wd["shadow_probs"], wd["shadow_threshold"], action="no_action"
                        ).decision_cost
                        c_prom_tot += c_p
                        c_shad_tot += c_s

                    if c_prom_tot > c_shad_tot + self.margin_cost:
                        ver = rollback_tracker["promoted_version"]
                        logger.warning(
                            "Rollback triggered at window %d for %s! Cost ($%.2f) exceeded shadow ($%.2f) by > $%.2f",
                            t, ver, c_prom_tot, c_shad_tot, self.margin_cost,
                        )
                        self.registry.rollback(
                            version=ver,
                            actor="rollback_monitor",
                            reason=f"Realised cost ${c_prom_tot:.2f} exceeded shadow ${c_shad_tot:.2f} by margin {self.margin_cost}",
                            metrics={
                                "promoted_cost": c_prom_tot,
                                "shadow_cost": c_shad_tot,
                                "evaluated_windows": eval_windows,
                            },
                        )
                        rolled_back_versions.append(ver)
                        # Revert champion
                        current_champion = rollback_tracker["shadow_model"]
                        current_decision_threshold = rollback_tracker["shadow_threshold"]
                        rollback_tracker = None
                    elif t >= rollback_tracker["expiry_window"] and all(
                        kw in eval_windows
                        for kw in range(rollback_tracker["promoted_window"] + 1, rollback_tracker["expiry_window"] + 1)
                    ):
                        logger.info("Promoted model %s passed shadow window safely.", rollback_tracker["promoted_version"])
                        rollback_tracker = None

            # 2. Performance metrics
            if n_fraud > 0:
                p_curve, r_curve, _ = precision_recall_curve(y_true, probs)
                pr_auc = float(auc(r_curve, p_curve))
                rec = float(((probs >= current_decision_threshold) & (y_true == 1)).sum() / n_fraud)
            else:
                pr_auc = 0.0
                rec = 1.0

            window_pr_aucs.append(pr_auc)
            window_recalls.append(rec)

            # 3. Update drift detectors and performance monitor
            det_results = {name: det.update(window) for name, det in detectors.items()}
            perf_result = performance_monitor.update(window) if performance_monitor is not None else None

            # 4. Form policy state and query decision
            state = PolicyState(
                window_index=t,
                current_detector_results=det_results,
                detector_history=detector_history,
                performance_result=perf_result,
                performance_history=performance_history,
                windows_since_last_retrain=windows_since_last_retrain,
                recent_costs=recent_costs,
                reference_cost_per_window=reference_cost_per_window,
            )

            decision = policy.decide(state)
            recorded_decisions.append({
                "window": t,
                "action": decision.action,
                "reason": decision.reason,
            })

            # 5. Evaluate financial cost
            cost_res = self.cost_model.evaluate_window(
                window_index=t,
                y_true=y_true,
                y_prob=probs,
                decision_threshold=current_decision_threshold,
                action=decision.action,
            )

            total_fn_fp += cost_res.decision_cost
            total_retrain += cost_res.retrain_cost
            total_alert += cost_res.alert_cost
            window_costs.append(cost_res.total_cost)
            recent_costs.append(cost_res.decision_cost)

            # 6. Log decision to LogStore
            if self.log_store is not None and self.log_decisions:
                self.log_store.append("policy_decisions", {
                    "scenario": scenario_name,
                    "window_index": t,
                    "policy": policy.name,
                    "action": decision.action,
                    "reason": decision.reason,
                    "signals": decision.signals,
                    "cost": cost_res.total_cost,
                })

            # 7. Execute retraining if requested
            if decision.action == "retrain":
                max_train_window_idx = t - self.label_delay
                available_stream_windows = [
                    w for w in stream_windows[:t+1]
                    if w.window_index <= max_train_window_idx
                ]

                # Strict assertion: no labels beyond t - label_delay
                assert all(w.window_index <= max_train_window_idx for w in available_stream_windows), (
                    f"Label leakage detected! Window {t} used future labels > {max_train_window_idx}"
                )

                if self.deployment_mode == "naive":
                    # Naive: train on all available stream windows, deploy immediately
                    if available_stream_windows:
                        s_feats = pd.concat([w.features_df for w in available_stream_windows], ignore_index=True)
                        s_labels = pd.concat([w.labels for w in available_stream_windows], ignore_index=True).values
                        X_s = self.preprocessor.transform(s_feats)
                        y_s = s_labels

                        w_s = np.where(y_s == 1, self.stream_weight, 1.0)
                        w_base = np.ones(len(self.y_train_base))

                        X_all = np.vstack([self.X_train_base, X_s])
                        y_all = np.concatenate([self.y_train_base, y_s])
                        w_all = np.concatenate([w_base, w_s])
                    else:
                        X_all = self.X_train_base
                        y_all = self.y_train_base
                        w_all = np.ones(len(self.y_train_base))

                    spw = float((w_all[y_all == 0].sum()) / max(1, w_all[y_all == 1].sum()))
                    challenger = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        scale_pos_weight=spw,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=-1,
                        random_state=self.seed + t,
                    )
                    challenger.fit(X_all, y_all, sample_weight=w_all)
                    current_champion = challenger
                    windows_since_last_retrain = 0
                    retrain_windows.append(t)

                elif self.deployment_mode == "governed":
                    # 1. Hold out the most recent K_val labelled windows as validation
                    if len(available_stream_windows) > self.k_val_windows:
                        val_stream_windows = available_stream_windows[-self.k_val_windows:]
                        train_stream_windows = available_stream_windows[:-self.k_val_windows]
                    elif len(available_stream_windows) > 1:
                        k = min(self.k_val_windows, len(available_stream_windows) - 1)
                        val_stream_windows = available_stream_windows[-k:]
                        train_stream_windows = available_stream_windows[:-k]
                    else:
                        val_stream_windows = available_stream_windows
                        train_stream_windows = []

                    val_indices = {w.window_index for w in val_stream_windows}
                    assert not any(w.window_index in val_indices for w in train_stream_windows), (
                        "Holdout window leaked into training set!"
                    )

                    # 2. Train challenger on earlier labelled windows + base sample
                    if train_stream_windows:
                        s_feats = pd.concat([w.features_df for w in train_stream_windows], ignore_index=True)
                        s_labels = pd.concat([w.labels for w in train_stream_windows], ignore_index=True).values
                        X_s = self.preprocessor.transform(s_feats)
                        y_s = s_labels

                        w_s = np.where(y_s == 1, self.stream_weight, 1.0)
                        w_base = np.ones(len(self.y_train_base))

                        X_all = np.vstack([self.X_train_base, X_s])
                        y_all = np.concatenate([self.y_train_base, y_s])
                        w_all = np.concatenate([w_base, w_s])
                    else:
                        X_all = self.X_train_base
                        y_all = self.y_train_base
                        w_all = np.ones(len(self.y_train_base))

                    spw = float((w_all[y_all == 0].sum()) / max(1, w_all[y_all == 1].sum()))
                    challenger = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        scale_pos_weight=spw,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=-1,
                        random_state=self.seed + t,
                    )
                    challenger.fit(X_all, y_all, sample_weight=w_all)

                    model_version_counter += 1
                    cand_version = f"v{model_version_counter}.0"
                    data_hash = f"train_win_{[w.window_index for w in train_stream_windows]}_rows_{len(y_all)}"

                    # 3. Validation and threshold recalibration on holdout
                    if val_stream_windows:
                        val_feats = pd.concat([w.features_df for w in val_stream_windows], ignore_index=True)
                        val_labels = pd.concat([w.labels for w in val_stream_windows], ignore_index=True).values
                        X_val = self.preprocessor.transform(val_feats)
                        y_val = val_labels
                        val_legit_idx = np.where(y_val == 0)[0]
                        val_fraud_idx = np.where(y_val == 1)[0]
                        probs_chall = challenger.predict_proba(X_val)[:, 1]
                        probs_champ = current_champion.predict_proba(X_val)[:, 1]
                    else:
                        val_legit_idx = np.array([])
                        val_fraud_idx = np.array([])
                        probs_chall = np.array([])
                        probs_champ = np.array([])
                        y_val = np.array([])

                    # Threshold recalibration to match FP budget on holdout
                    if len(val_legit_idx) < 20:
                        challenger_threshold = current_decision_threshold
                        champion_threshold = current_decision_threshold
                    else:
                        q = max(0.0, min(1.0, 1.0 - fp_budget))
                        challenger_threshold = float(np.quantile(probs_chall[val_legit_idx], q))
                        champion_threshold = float(np.quantile(probs_champ[val_legit_idx], q))

                    # Compute validation metrics on holdout
                    if len(val_fraud_idx) > 0:
                        chall_rec = float(((probs_chall[val_fraud_idx] >= challenger_threshold)).sum() / len(val_fraud_idx))
                        champ_rec = float(((probs_champ[val_fraud_idx] >= champion_threshold)).sum() / len(val_fraud_idx))
                        p_ch, r_ch, _ = precision_recall_curve(y_val, probs_chall)
                        chall_pr_auc = float(auc(r_ch, p_ch))
                        p_cp, r_cp, _ = precision_recall_curve(y_val, probs_champ)
                        champ_pr_auc = float(auc(r_cp, p_cp))
                    else:
                        chall_rec = 0.0
                        champ_rec = 0.0
                        chall_pr_auc = 0.0
                        champ_pr_auc = 0.0

                    gate_metrics = {
                        "challenger_recall": chall_rec,
                        "champion_recall": champ_rec,
                        "challenger_pr_auc": chall_pr_auc,
                        "champion_pr_auc": champ_pr_auc,
                        "challenger_threshold": challenger_threshold,
                        "champion_threshold": champion_threshold,
                        "fp_budget": fp_budget,
                        "holdout_windows": [w.window_index for w in val_stream_windows],
                    }

                    # Register candidate
                    self.registry.register_candidate(
                        model=challenger,
                        version=cand_version,
                        threshold=challenger_threshold,
                        data_hash=data_hash,
                        metrics=gate_metrics,
                        actor="governance_pipeline",
                        reason=f"Candidate trained on labelled windows <= {max_train_window_idx}",
                        created_window=t,
                    )

                    # 4. Promotion Gate:
                    # Promote only if challenger recall at FP budget beats champion recall
                    # by margin_recall AND challenger PR-AUC is not lower.
                    passes_gate = (
                        (chall_rec >= champ_rec + self.margin_recall)
                        and (chall_pr_auc >= champ_pr_auc)
                    )

                    if passes_gate:
                        self.registry.promote(
                            version=cand_version,
                            actor="promotion_gate",
                            reason=(
                                f"Challenger recall ({chall_rec:.3f}) >= champ ({champ_rec:.3f}) + {self.margin_recall} "
                                f"and PR-AUC ({chall_pr_auc:.3f} >= {champ_pr_auc:.3f})"
                            ),
                            metrics=gate_metrics,
                        )
                        promoted_versions.append(cand_version)
                        shadow_champion = current_champion
                        shadow_threshold = current_decision_threshold
                        current_champion = challenger
                        current_decision_threshold = challenger_threshold
                        windows_since_last_retrain = 0
                        retrain_windows.append(t)

                        # Set up shadow rollback tracking for next M windows
                        rollback_tracker = {
                            "promoted_version": cand_version,
                            "promoted_window": t,
                            "shadow_model": shadow_champion,
                            "shadow_threshold": shadow_threshold,
                            "expiry_window": t + self.rollback_window_m,
                            "window_data": {},
                        }
                    else:
                        self.registry.reject(
                            version=cand_version,
                            actor="promotion_gate",
                            reason=(
                                f"Challenger failed gate (rec: {chall_rec:.3f} vs {champ_rec:.3f}+{self.margin_recall}, "
                                f"pr_auc: {chall_pr_auc:.3f} vs {champ_pr_auc:.3f})"
                            ),
                            metrics=gate_metrics,
                        )
                        rejected_versions.append(cand_version)
                        # Current champion continues untouched; retrain cost is paid
            else:
                windows_since_last_retrain += 1

            # Update histories
            detector_history.append(det_results)
            if perf_result:
                performance_history.append(perf_result)

        duration = time.time() - start_time

        # Calculate detection delay and false alarms
        action_windows = [
            d["window"] for d in recorded_decisions
            if d["action"] in {"alert", "retrain"}
        ]

        if scenario_name == "control":
            false_alarms = len(action_windows)
            detection_delay = None
        else:
            false_alarms = len([w for w in action_windows if w < onset_window])
            post_actions = [w for w in action_windows if w >= onset_window]
            detection_delay = (post_actions[0] - onset_window) if post_actions else None

        post_onset_aucs = window_pr_aucs[onset_window:]
        mean_pr_auc_post = float(np.mean(post_onset_aucs)) if post_onset_aucs else 0.0

        return ReplayResult(
            scenario=scenario_name,
            policy=policy.name,
            seed=self.seed,
            onset_window=onset_window,
            n_retrains=len(retrain_windows),
            retrain_windows=retrain_windows,
            detection_delay=detection_delay,
            false_alarms=false_alarms,
            total_fn_fp_cost=total_fn_fp,
            total_retrain_cost=total_retrain,
            total_alert_cost=total_alert,
            total_cost=total_fn_fp + total_retrain + total_alert,
            mean_pr_auc_after_onset=mean_pr_auc_post,
            duration_seconds=round(duration, 2),
            promoted_versions=promoted_versions,
            rejected_versions=rejected_versions,
            rolled_back_versions=rolled_back_versions,
            per_window_costs=window_costs,
            per_window_pr_auc=window_pr_aucs,
            per_window_recall=window_recalls,
            decisions=recorded_decisions,
        )
