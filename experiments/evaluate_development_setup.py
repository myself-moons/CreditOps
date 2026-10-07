"""
experiments/evaluate_development_setup.py — CreditOps v2 Phase 3 Evaluation

Runs:
  1. Forced retrain verification at window 22 for concept_drift, novel_pattern, covariate_shift.
     Reports pooled PR-AUC and Recall for windows 25-34 for (no retrain) vs (retrain).
  2. Full policy comparison on the development setup (onset 15, seed 0):
     5 scenarios x 4 policies -> n_retrains, detection_delay, false_alarms, total_cost,
     loss_avoided, net_benefit, seconds.
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_adapter import DatasetAdapter
from src.features import build_features
from src.observatory.config import DriftInjectionConfig, load_config
from src.observatory.cost import CostModel
from src.observatory.monitoring.detectors import (
    DataDriftDetector,
    FraudScoreDriftDetector,
    IsolationOutlierDetector,
    UnseenCategoryDetector,
)
from src.observatory.monitoring.performance import PerformanceMonitor
from src.observatory.policies import (
    AdaptivePolicy,
    FixedSchedule,
    NeverRetrain,
    ThresholdPolicy,
)
from src.observatory.simulator.engine import ReplayEngine, ReplayResult
from src.observatory.simulator.stream import WindowedStream
from src.observatory.storage.sqlite_store import SQLiteLogStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_dev")


def run_forced_retrain_study(
    raw_slice: pd.DataFrame,
    champion_model: Any,
    preprocessor: Any,
    X_train_base: np.ndarray,
    y_train_base: np.ndarray,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Forces a retrain at window 22 under concept_drift, novel_pattern, and covariate_shift
    (onset 15, seed 0). Compares pooled PR-AUC and recall on windows 25-34 for (no retrain) vs (retrain).
    """
    logger.info("=== RUNNING FORCED RETRAIN STUDY (ITEM E) ===")
    results = []

    scenarios = [
        ("concept_drift", 1.0),
        ("novel_pattern", 1.0),
        ("covariate_shift", 1.5),
    ]

    for sc_name, mag in scenarios:
        cfg = load_config()
        cfg.seed = seed
        cfg.drift_injection = DriftInjectionConfig(
            scenario=sc_name,
            shape="sudden",
            onset_window=15,
            magnitude=mag,
        )

        stream = list(WindowedStream(raw_df=raw_slice, config=cfg))

        # Windows 25-34 evaluation slice
        eval_windows = stream[25:35]
        ef = pd.concat([w.features_df for w in eval_windows], ignore_index=True)
        el = pd.concat([w.labels for w in eval_windows], ignore_index=True).values

        # 1. Baseline: No Retrain
        probs_base = champion_model.predict_proba(preprocessor.transform(ef))[:, 1]
        p_b, r_b, _ = precision_recall_curve(el, probs_base)
        auc_base = auc(r_b, p_b)
        rec_base = ((probs_base >= 0.97) & (el == 1)).sum() / max(1, el.sum())

        # 2. Retrain at window 22 using labelled windows <= 20 (delay = 2)
        train_stream_windows = [w for w in stream[:23] if w.window_index <= 20]
        assert max(w.window_index for w in train_stream_windows) <= 20

        s_feats = pd.concat([w.features_df for w in train_stream_windows], ignore_index=True)
        s_labels = pd.concat([w.labels for w in train_stream_windows], ignore_index=True).values
        X_s = preprocessor.transform(s_feats)
        y_s = s_labels

        w_s = np.where(y_s == 1, 20.0, 1.0)
        w_base = np.ones(len(y_train_base))

        X_comb = np.vstack([X_train_base, X_s])
        y_comb = np.concatenate([y_train_base, y_s])
        w_comb = np.concatenate([w_base, w_s])

        spw = float((w_comb[y_comb == 0].sum()) / max(1, w_comb[y_comb == 1].sum()))
        clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=spw,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_comb, y_comb, sample_weight=w_comb)

        probs_ret = clf.predict_proba(preprocessor.transform(ef))[:, 1]
        p_r, r_r, _ = precision_recall_curve(el, probs_ret)
        auc_ret = auc(r_r, p_r)
        rec_ret = ((probs_ret >= 0.97) & (el == 1)).sum() / max(1, el.sum())

        results.append({
            "Scenario": sc_name,
            "No Retrain PR-AUC": round(float(auc_base), 4),
            "Retrained PR-AUC": round(float(auc_ret), 4),
            "PR-AUC Change": f"{float(auc_ret - auc_base):+.4f}",
            "No Retrain Recall": round(float(rec_base), 4),
            "Retrained Recall": round(float(rec_ret), 4),
            "Recall Change": f"{float(rec_ret - rec_base):+.4f}",
        })

    df_retrain = pd.DataFrame(results)
    logger.info("\n%s\n", df_retrain.to_string(index=False))
    return df_retrain


def run_development_policy_study(
    raw_slice: pd.DataFrame,
    champion_model: Any,
    preprocessor: Any,
    X_train_base: np.ndarray,
    y_train_base: np.ndarray,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Runs full policy comparison on development setup (onset 15, seed 0).
    5 scenarios x 4 policies -> n_retrains, detection_delay, false_alarms,
    total_cost, loss_avoided, net_benefit, seconds.
    """
    logger.info("=== RUNNING DEVELOPMENT SETUP POLICY COMPARISON (ITEM F) ===")

    cfg = load_config()
    cost_model = CostModel(
        fn_cost=cfg.cost_model.fn_cost,
        fp_cost=cfg.cost_model.fp_cost,
        retrain_cost=cfg.cost_model.retrain_cost,
        review_cost=cfg.cost_model.review_cost,
    )

    # Reference sample on control stream
    ctrl_cfg = load_config()
    ctrl_cfg.seed = seed
    ctrl_cfg.drift_injection = DriftInjectionConfig(scenario="control", shape="sudden", onset_window=15)
    ctrl_stream = list(WindowedStream(raw_df=raw_slice, config=ctrl_cfg))

    r_count = cfg.monitoring.reference_windows
    ref_df = pd.concat([w.features_df for w in ctrl_stream[:r_count]], ignore_index=True)
    ref_labels = pd.concat([w.labels for w in ctrl_stream[:r_count]], ignore_index=True).values
    ref_scores = champion_model.predict_proba(preprocessor.transform(ref_df))[:, 1]

    prec_c, rec_c, _ = precision_recall_curve(ref_labels, ref_scores)
    ref_pr_auc = float(auc(rec_c, prec_c))

    # Reference window cost
    ref_win_costs = []
    for w in ctrl_stream[:r_count]:
        y_w = w.labels.values
        p_w = champion_model.predict_proba(preprocessor.transform(w.features_df))[:, 1]
        c_res = cost_model.evaluate_window(w.window_index, y_w, p_w, decision_threshold=0.97)
        ref_win_costs.append(c_res.decision_cost)
    ref_cost_per_window = float(np.mean(ref_win_costs))

    # Fit IsolationForest on random ~50k train sample
    from sklearn.ensemble import IsolationForest
    iso_model = IsolationForest(n_estimators=100, random_state=seed, contamination=0.05)
    iso_model.fit(X_train_base)

    scenarios = [
        ("control", 1.0),
        ("covariate_shift", 1.5),
        ("fraud_rate_shift", 3.0),
        ("concept_drift", 1.0),
        ("novel_pattern", 1.0),
    ]

    log_store = SQLiteLogStore(db_path=str(ROOT / "runs" / "dev_policy_decisions.db"))

    all_results: List[ReplayResult] = []

    for sc_name, mag in scenarios:
        logger.info("Evaluating scenario: %s...", sc_name)
        sc_cfg = load_config()
        sc_cfg.seed = seed
        sc_cfg.drift_injection = DriftInjectionConfig(
            scenario=sc_name,
            shape="sudden",
            onset_window=15,
            magnitude=mag,
        )
        stream_windows = list(WindowedStream(raw_df=raw_slice, config=sc_cfg))

        policy_configs = [
            (NeverRetrain(), "naive", "NeverRetrain"),
            (FixedSchedule(k_windows=20), "naive", "FixedSchedule(naive)"),
            (ThresholdPolicy(cooldown_windows=5), "naive", "ThresholdPolicy(naive)"),
            (
                AdaptivePolicy(
                    n_perf_persistence=2,
                    n_free_persistence=2,
                    cooldown_windows=5,
                    horizon_windows=10,
                    retrain_cost=cfg.cost_model.retrain_cost,
                ),
                "naive",
                "AdaptivePolicy(naive)",
            ),
            (
                AdaptivePolicy(
                    n_perf_persistence=2,
                    n_free_persistence=2,
                    cooldown_windows=5,
                    horizon_windows=10,
                    retrain_cost=cfg.cost_model.retrain_cost,
                ),
                "governed",
                "AdaptivePolicy(governed)",
            ),
        ]

        never_fn_fp_cost = 0.0

        for pol, dep_mode, pol_label in policy_configs:
            # Fresh detector instances
            detectors = {
                "DataDrift": DataDriftDetector(
                    reference_df=ref_df,
                    k_prev=cfg.monitoring.k_prev_windows,
                    threshold=cfg.monitoring.threshold_data_drift,
                    log_store=log_store,
                ),
                "UnseenCategory": UnseenCategoryDetector(
                    min_unseen_rows=cfg.monitoring.threshold_unseen_category,
                    log_store=log_store,
                ),
                "IsolationOutlier": IsolationOutlierDetector(
                    isolation_forest=iso_model,
                    preprocessor=preprocessor,
                    threshold=cfg.monitoring.threshold_isolation_outlier,
                    log_store=log_store,
                ),
                "FraudScoreDrift": FraudScoreDriftDetector(
                    champion_model=champion_model,
                    preprocessor=preprocessor,
                    reference_scores=ref_scores,
                    threshold=cfg.monitoring.threshold_fraud_score_drift,
                    log_store=log_store,
                ),
            }
            perf_monitor = PerformanceMonitor(
                champion_model=champion_model,
                preprocessor=preprocessor,
                label_delay_windows=cfg.monitoring.label_delay_windows,
                k_perf=cfg.monitoring.k_perf_windows,
                min_fraud_rows=cfg.monitoring.min_fraud_rows,
                decision_threshold=0.97,
                threshold_recall=cfg.monitoring.threshold_performance_recall,
                reference_pr_auc=ref_pr_auc,
                threshold_pr_auc_drop=cfg.monitoring.threshold_pr_auc_drop,
                log_store=log_store,
            )

            engine = ReplayEngine(
                champion_model=champion_model,
                preprocessor=preprocessor,
                cost_model=cost_model,
                base_train_features=X_train_base,
                base_train_labels=y_train_base,
                decision_threshold=0.97,
                label_delay_windows=cfg.monitoring.label_delay_windows,
                stream_weight=20.0,
                deployment_mode=dep_mode,
                log_store=log_store,
                log_decisions=True,
                seed=seed,
                k_val_windows=cfg.governance.k_val_windows,
                margin_recall=cfg.governance.margin_recall,
                margin_cost=cfg.governance.margin_cost,
                rollback_window_m=cfg.governance.rollback_window_m,
            )

            res = engine.run(
                stream_windows=stream_windows,
                policy=pol,
                detectors=detectors,
                performance_monitor=perf_monitor,
                scenario_name=sc_name,
                onset_window=15,
                reference_cost_per_window=ref_cost_per_window,
            )
            res.policy = pol_label

            if pol_label == "NeverRetrain":
                never_fn_fp_cost = res.total_fn_fp_cost
                res.loss_avoided_vs_never = 0.0
                res.net_benefit = 0.0
            else:
                res.loss_avoided_vs_never = never_fn_fp_cost - res.total_fn_fp_cost
                res.net_benefit = res.loss_avoided_vs_never - res.total_retrain_cost

            all_results.append(res)

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Scenario": r.scenario,
            "Policy": r.policy,
            "Retrains": r.n_retrains,
            "Delay": r.detection_delay if r.detection_delay is not None else "—",
            "False Alarms": r.false_alarms,
            "Total Cost ($)": f"${r.total_cost:,.2f}",
            "Loss Avoided ($)": f"${r.loss_avoided_vs_never:,.2f}",
            "Net Benefit ($)": f"${r.net_benefit:,.2f}",
            "Post PR-AUC": f"{r.mean_pr_auc_after_onset:.4f}",
            "Time (s)": f"{r.duration_seconds:.1f}s",
        })

    df_summary = pd.DataFrame(summary_rows)
    logger.info("\n=== DEVELOPMENT SETUP RESULTS (5 SCENARIOS x 4 POLICIES) ===\n%s\n", df_summary.to_string(index=False))
    return df_summary


def main():
    cfg = load_config()

    with open(ROOT / "model.pkl", "rb") as f:
        champion = pickle.load(f)
    with open(ROOT / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    # Pre-sample training data once
    logger.info("Sampling 50k rows from fraudTrain.csv...")
    chunks = []
    for chunk in pd.read_csv(ROOT / cfg.raw_train, chunksize=250000, low_memory=False):
        chunks.append(chunk.sample(n=min(len(chunk), 10000), random_state=cfg.seed))
    df_sample = pd.concat(chunks, ignore_index=True)
    X_train_base = preprocessor.transform(build_features(df_sample))
    y_train_base = df_sample["is_fraud"].values

    # Pre-load raw slice once
    adapter = DatasetAdapter(train_path=cfg.raw_train, test_path=cfg.raw_test)
    raw_slice = adapter.load_slice(cfg.stream_start, cfg.stream_end)

    # 1. Run forced retrain study
    df_retrain = run_forced_retrain_study(
        raw_slice=raw_slice,
        champion_model=champion,
        preprocessor=preprocessor,
        X_train_base=X_train_base,
        y_train_base=y_train_base,
        seed=0,
    )

    # 2. Run policy comparison study
    df_policies = run_development_policy_study(
        raw_slice=raw_slice,
        champion_model=champion,
        preprocessor=preprocessor,
        X_train_base=X_train_base,
        y_train_base=y_train_base,
        seed=0,
    )


if __name__ == "__main__":
    main()
