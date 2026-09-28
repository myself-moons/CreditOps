"""
model_training.py — CreditOps

DVC Stage: Model_Training
--------------------------
MODEL SELECTION METHODOLOGY
----------------------------
The pipeline uses a temporal split (70/15/15) to avoid data leakage.

  Training data (70%)
      |
      |-- Logistic Regression   → cross-val PR-AUC on training folds
      |-- Random Forest         → cross-val PR-AUC on training folds
      |-- XGBoost               → cross-val PR-AUC on training folds
      |-- LightGBM (if available)
      |
      |  Select model by mean CV PR-AUC (not ROC-AUC — fraud is ~0.5%)
      |  Retrain selected model on FULL training split
      |
  Validation data (15%) → Tune decision threshold
  Test data (15%)        → UNTOUCHED until final evaluation

PRIMARY METRIC: PR-AUC (average precision)
  At ~0.5% fraud rate, accuracy and ROC-AUC are misleading.
  PR-AUC focuses on the minority class and is the correct selection metric.

IMBALANCE HANDLING
  Majority-class downsampling for training only (documented in params.yaml).
  Full val and test splits are always evaluated without downsampling.

MLflow experiment: credit-fraud
Model Registry:    champion alias set on the best model.

SIMULATED DATA NOTICE: Source data is Sparkov simulation.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "is_fraud"
CV_N_SPLITS = 3   # fewer folds for speed on large data


def load_params() -> dict:
    with (ROOT_DIR / "params.yaml").open() as f:
        return yaml.safe_load(f)


def downsample_majority(X: pd.DataFrame, y: np.ndarray, frac: float, rng: int) -> tuple:
    """Downsample majority class (non-fraud) for TRAINING ONLY."""
    idx_fraud    = np.where(y == 1)[0]
    idx_nonfraud = np.where(y == 0)[0]
    rng_obj      = np.random.default_rng(rng)
    keep_n       = max(1, int(len(idx_nonfraud) * frac))
    keep_idx     = rng_obj.choice(idx_nonfraud, size=keep_n, replace=False)
    all_idx      = np.concatenate([idx_fraud, keep_idx])
    rng_obj.shuffle(all_idx)
    X_out = X.iloc[all_idx].reset_index(drop=True)
    y_out = y[all_idx]
    logger.info("Downsampled majority: %d non-fraud kept (%.0f%%) + %d fraud = %d total",
                keep_n, frac * 100, len(idx_fraud), len(all_idx))
    return X_out, y_out


def compute_metrics(y_true, y_prob, threshold: float = 0.5, fp_budget: float = 0.05) -> dict:
    """Compute full metric suite including PR-AUC and recall at FP budget."""
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_prob)
    roc    = roc_auc_score(y_true, y_prob)

    # Recall at fixed false-positive rate budget
    n_neg = int((y_true == 0).sum())
    fp_budget_count = int(n_neg * fp_budget)
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
    # Compute FP at each threshold
    recall_at_budget = 0.0
    for t in sorted(np.unique(y_prob), reverse=True):
        y_p = (y_prob >= t).astype(int)
        fp_count = int(((y_p == 1) & (y_true == 0)).sum())
        if fp_count <= fp_budget_count:
            recall_at_budget = float(recall_score(y_true, y_p, zero_division=0))
            break

    return {
        "pr_auc":            round(pr_auc, 6),
        "roc_auc":           round(roc, 6),
        "accuracy":          round(accuracy_score(y_true, y_pred), 6),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall":            round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1_score":          round(f1_score(y_true, y_pred, zero_division=0), 6),
        "recall_at_fp_budget": round(recall_at_budget, 6),
        "fp_budget_rate":    fp_budget,
        "threshold":         round(threshold, 4),
    }


def tune_threshold(y_true, y_prob, metric: str = "f1") -> float:
    """Find threshold that maximises F1 on val set."""
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.01, 0.99, 0.01):
        y_p = (y_prob >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, y_p, zero_division=0)
        else:
            s = average_precision_score(y_true, y_p)
        if s > best_s:
            best_s, best_t = s, t
    logger.info("Best threshold on val: %.2f  (val F1=%.4f)", best_t, best_s)
    return float(best_t)


def build_model(model_type: str, params: dict, scale_pos_weight: float = 1.0):
    rs = params["model"]["random_state"]
    ne = params["model"]["n_estimators"]
    md = params["model"]["max_depth"]
    lr = params["model"]["learning_rate"]
    ss = params["model"]["subsample"]
    cb = params["model"]["colsample_bytree"]
    C  = params["model"]["C"]
    mi = params["model"]["max_iter"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=C, max_iter=mi, random_state=rs,
            class_weight="balanced", solver="lbfgs", n_jobs=-1,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=ne, max_depth=md, random_state=rs,
            class_weight="balanced", n_jobs=-1,
        )
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=ss, colsample_bytree=cb, random_state=rs,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", n_jobs=-1, verbosity=0,
        )
    if model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=ss, colsample_bytree=cb, random_state=rs,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1, verbosity=-1,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def main() -> None:
    params = load_params()
    m_cfg  = params["model"]
    d_cfg  = params["data"]
    mon_cfg = params["monitoring"]
    fp_budget = float(mon_cfg.get("fp_budget_rate", 0.05))
    ds_frac   = float(d_cfg.get("majority_downsample_frac", 0.30))
    rs        = int(m_cfg["random_state"])

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT_DIR / 'mlruns'}")
    )
    mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", params["mlflow"]["experiment_name"])
    )

    # ── Load processed data ─────────────────────────────────────────────── #
    proc_dir = ROOT_DIR / "data" / "processed"
    train = pd.read_csv(proc_dir / "train_processed.csv")
    val   = pd.read_csv(proc_dir / "val_processed.csv")
    test  = pd.read_csv(proc_dir / "test_processed.csv")

    X_train_full = train.drop(columns=[TARGET])
    y_train_full = train[TARGET].values
    X_val  = val.drop(columns=[TARGET])
    y_val  = val[TARGET].values
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].values

    # ── Downsample majority class for training (documented) ─────────────── #
    logger.info("Downsampling majority class (non-fraud) in TRAIN for speed.")
    logger.info("Val and test splits are NOT downsampled — full evaluation.")
    X_train, y_train = downsample_majority(X_train_full, y_train_full, ds_frac, rs)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    logger.info("Train after downsampling: %d rows | neg=%d pos=%d | scale_pos_weight=%.2f",
                len(y_train), neg, pos, scale_pos_weight)

    # ── Model candidates ────────────────────────────────────────────────── #
    mt = m_cfg["model_type"]
    candidates = (
        ["logistic_regression", "random_forest", "xgboost"]
        if mt == "all"
        else [mt]
    )

    # Try lightgbm — include only if it installs cleanly
    try:
        import lightgbm  # noqa: F401
        if mt == "all":
            candidates.append("lightgbm")
        logger.info("LightGBM available — adding to candidates")
    except ImportError:
        logger.info("LightGBM not installed — skipping")

    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=rs)

    # ── Phase 1: Cross-validate on training data ─────────────────────────── #
    logger.info("=== Phase 1: %d-fold CV on training data (PR-AUC) ===", CV_N_SPLITS)
    cv_results: dict[str, dict] = {}

    for mt_name in candidates:
        clf = build_model(mt_name, params, scale_pos_weight)
        scores = cross_val_score(
            clf, X_train, y_train, cv=skf,
            scoring="average_precision", n_jobs=1,
        )
        mean_pr = float(np.mean(scores))
        std_pr  = float(np.std(scores))
        logger.info("  [%s]  CV PR-AUC = %.4f ± %.4f  folds=%s",
                    mt_name, mean_pr, std_pr, [round(s, 4) for s in scores])

        with mlflow.start_run(run_name=f"{mt_name}_cv") as run:
            mlflow.set_tags({
                "run_stage": "cv_candidate",
                "model_family": mt_name,
                "data_provenance": "synthetic/simulated (Sparkov)",
            })
            mlflow.log_params({"model_type": mt_name, **{
                k: v for k, v in m_cfg.items() if k != "model_type"
            }})
            mlflow.log_metrics({
                "cv_mean_pr_auc": mean_pr,
                "cv_std_pr_auc": std_pr,
                **{f"cv_fold_{i+1}_pr_auc": float(s) for i, s in enumerate(scores)},
            })
            cv_run_id = run.info.run_id

        cv_results[mt_name] = {
            "mean_pr": mean_pr, "std_pr": std_pr,
            "fold_scores": scores.tolist(), "cv_run_id": cv_run_id,
        }

    # ── Phase 2: Select best model ──────────────────────────────────────── #
    best_name = max(cv_results, key=lambda m: cv_results[m]["mean_pr"])
    best_cv   = cv_results[best_name]
    logger.info("=== Phase 2: Selected '%s' (CV PR-AUC=%.4f) ===", best_name, best_cv["mean_pr"])

    # ── Phase 3: Retrain on full training split ─────────────────────────── #
    logger.info("Retraining on FULL training split...")
    final_clf = build_model(best_name, params, scale_pos_weight)
    final_clf.fit(X_train, y_train)

    # ── Phase 4: Tune threshold on VAL (not test) ──────────────────────── #
    val_prob  = final_clf.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_prob, metric="f1")

    val_m = compute_metrics(y_val, val_prob, threshold=threshold, fp_budget=fp_budget)
    logger.info("Val metrics: PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f recall=%.4f recall@FPR=%.4f",
                val_m["pr_auc"], val_m["roc_auc"], val_m["f1_score"],
                val_m["recall"], val_m["recall_at_fp_budget"])

    # ── Phase 5: Evaluate ONCE on untouched test split ─────────────────── #
    logger.info("=== Phase 5: Final evaluation on UNTOUCHED test split ===")
    test_prob = final_clf.predict_proba(X_test)[:, 1]
    test_m    = compute_metrics(y_test, test_prob, threshold=threshold, fp_budget=fp_budget)
    logger.info("Test metrics: PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f recall=%.4f recall@FPR=%.4f",
                test_m["pr_auc"], test_m["roc_auc"], test_m["f1_score"],
                test_m["recall"], test_m["recall_at_fp_budget"])

    # ── Phase 6: Log final model to MLflow ─────────────────────────────── #
    with mlflow.start_run(run_name=f"{best_name}_final") as final_run:
        mlflow.set_tags({
            "run_stage": "final_champion",
            "model_family": best_name,
            "selected_as_champion": "true",
            "data_provenance": "synthetic/simulated (Sparkov)",
        })
        mlflow.log_params({
            "model_type": best_name,
            "selection_metric": "pr_auc",
            "threshold": threshold,
            "downsample_frac": ds_frac,
            **{k: v for k, v in m_cfg.items() if k != "model_type"},
        })
        mlflow.log_metrics({
            "cv_mean_pr_auc": best_cv["mean_pr"],
            "cv_std_pr_auc":  best_cv["std_pr"],
            **{f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))},
            **{f"test_{k}": v for k, v in test_m.items() if isinstance(v, (int, float))},
        })
        mlflow.sklearn.log_model(final_clf, "model")
        final_run_id = final_run.info.run_id

        # Register in MLflow Model Registry with champion alias
        try:
            model_uri = f"runs:/{final_run_id}/model"
            reg = mlflow.register_model(model_uri, "credit-fraud-classifier")
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                "credit-fraud-classifier",
                params["mlflow"].get("champion_alias", "champion"),
                reg.version,
            )
            logger.info("Registered as champion v%s", reg.version)
        except Exception as e:
            logger.warning("Model Registry step failed (non-fatal): %s", e)

    # ── Save artifacts ──────────────────────────────────────────────────── #
    (ROOT_DIR / ".mlflow_run_id").write_text(final_run_id)

    with (ROOT_DIR / "model.pkl").open("wb") as f:
        pickle.dump(final_clf, f)

    # Save threshold for use by API and evaluation
    (ROOT_DIR / ".decision_threshold").write_text(str(threshold))

    logger.info("model.pkl + .mlflow_run_id + .decision_threshold saved.")
    logger.info("Training complete. Champion: %s  Test PR-AUC: %.4f",
                best_name, test_m["pr_auc"])


if __name__ == "__main__":
    main()