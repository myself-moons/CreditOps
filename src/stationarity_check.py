"""
stationarity_check.py — CreditOps

DVC Stage: Stationarity_Check
--------------------------------
Measures how stationary the (simulated) data really is.

Per calendar month and per split, computes:
  - fraud_rate
  - PSI and KS statistic for each model feature vs training baseline
  - Champion model PR-AUC and recall-at-budget on val and test periods

Outputs:
  - stationarity_report.json
  - stationarity_report.csv

IMPORTANT: Any drift found in this dataset is NATURAL variation within
the Sparkov simulation — not real-world concept drift. Later capstone
drift experiments must be explicitly injected, seeded, and labelled
"injected" so detection delay can be measured.

SIMULATED DATA NOTICE: Source data is Sparkov simulation.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import average_precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import build_features, feature_names
from src.dataset_adapter import DatasetAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "is_fraud"
N_BINS = 10   # PSI bins


def _psi(base: np.ndarray, curr: np.ndarray, n_bins: int = N_BINS) -> float:
    """Population Stability Index between base and curr distributions."""
    base = base[~np.isnan(base)]
    curr = curr[~np.isnan(curr)]
    if len(base) < 10 or len(curr) < 10:
        return float("nan")
    try:
        bins = np.percentile(base, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return float("nan")
        base_counts = np.histogram(base, bins=bins)[0]
        curr_counts = np.histogram(curr, bins=bins)[0]
        base_pct = (base_counts + 0.0001) / len(base)
        curr_pct = (curr_counts + 0.0001) / len(curr)
        psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
        return round(psi, 6)
    except Exception:
        return float("nan")


def _ks(base: np.ndarray, curr: np.ndarray) -> float:
    """KS statistic between base and current feature distributions."""
    try:
        stat, _ = scipy_stats.ks_2samp(base, curr)
        return round(float(stat), 6)
    except Exception:
        return float("nan")


def main() -> None:
    logger.info("=== CreditOps Stationarity Check ===")
    logger.info("NOTICE: This dataset is simulated (Sparkov). Any variation observed is")
    logger.info("        natural simulation variation, NOT real-world concept drift.")

    # ── Load model + preprocessor ──────────────────────────────────────── #
    model_path = ROOT_DIR / "model.pkl"
    preproc_path = ROOT_DIR / "preprocessor.pkl"
    if not model_path.exists() or not preproc_path.exists():
        logger.error("model.pkl or preprocessor.pkl not found. Run dvc repro first.")
        sys.exit(1)

    with model_path.open("rb") as f:
        model = pickle.load(f)
    with preproc_path.open("rb") as f:
        preprocessor = pickle.load(f)

    threshold_path = ROOT_DIR / ".decision_threshold"
    threshold = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5

    fp_budget = 0.05
    adapter = DatasetAdapter()

    # ── Load train split as baseline ─────────────────────────────────────── #
    train_path = ROOT_DIR / "data" / "raw" / "train.csv"
    val_path   = ROOT_DIR / "data" / "raw" / "val.csv"
    test_path  = ROOT_DIR / "data" / "raw" / "test.csv"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        logger.error("Raw splits not found. Run dvc repro Data_Collection first.")
        sys.exit(1)

    logger.info("Loading splits...")
    train_raw = pd.read_csv(train_path, low_memory=False)
    val_raw   = pd.read_csv(val_path,   low_memory=False)
    test_raw  = pd.read_csv(test_path,  low_memory=False)

    # Build feature matrices for PSI/KS baseline
    logger.info("Building feature matrices...")
    X_train_feat = build_features(train_raw)
    feat_names   = feature_names()

    # Training baseline statistics (per feature)
    train_baseline = {col: X_train_feat[col].values for col in feat_names}
    train_fr = float(train_raw[TARGET].mean())
    logger.info("Training fraud rate: %.6f", train_fr)

    # ── Monthly analysis ─────────────────────────────────────────────────── #
    logger.info("Running monthly analysis...")
    monthly_rows = []

    for split_name, split_df in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        split_df = split_df.copy()
        split_df["_period"] = pd.to_datetime(split_df[adapter.time_column]).dt.to_period("M")

        for period, group in split_df.groupby("_period", sort=True):
            if len(group) < 50:
                continue

            period_str = str(period)
            fr = float(group[TARGET].mean())
            n  = len(group)

            # Feature PSI and KS vs training baseline
            X_group = build_features(group)
            psi_vals = {f"psi_{col}": _psi(train_baseline[col], X_group[col].values)
                        for col in feat_names}
            ks_vals  = {f"ks_{col}": _ks(train_baseline[col], X_group[col].values)
                        for col in feat_names}

            # Model performance (only for val and test)
            pr_auc = None
            recall_at_budget = None
            if split_name in ("val", "test") and int(group[TARGET].sum()) >= 2:
                X_enc = preprocessor.transform(X_group)
                y_prob = model.predict_proba(X_enc)[:, 1]
                y_true = group[TARGET].values
                try:
                    pr_auc = round(float(average_precision_score(y_true, y_prob)), 6)
                    n_neg = int((y_true == 0).sum())
                    fp_budget_count = int(n_neg * fp_budget)
                    for t in sorted(np.unique(y_prob), reverse=True):
                        y_p = (y_prob >= t).astype(int)
                        if int(((y_p == 1) & (y_true == 0)).sum()) <= fp_budget_count:
                            recall_at_budget = round(float(recall_score(y_true, y_p, zero_division=0)), 6)
                            break
                except Exception as e:
                    logger.warning("PR-AUC failed for %s %s: %s", split_name, period_str, e)

            row = {
                "split": split_name,
                "period": period_str,
                "n_samples": n,
                "fraud_count": int(group[TARGET].sum()),
                "fraud_rate": round(fr, 6),
                "pr_auc": pr_auc,
                "recall_at_fp_budget": recall_at_budget,
                **psi_vals,
                **ks_vals,
            }
            monthly_rows.append(row)

    report_df = pd.DataFrame(monthly_rows)

    # ── Summary statistics ────────────────────────────────────────────────── #
    # PSI interpretation: <0.1 = stable, 0.1-0.2 = minor shift, >0.2 = major shift
    psi_cols = [c for c in report_df.columns if c.startswith("psi_")]
    mean_psi = report_df[psi_cols].mean().mean() if psi_cols else float("nan")

    val_pr  = report_df[report_df["split"] == "val"]["pr_auc"].dropna()
    test_pr = report_df[report_df["split"] == "test"]["pr_auc"].dropna()

    # Honest conclusion based on actual numbers
    if not np.isnan(mean_psi):
        if mean_psi < 0.1:
            conclusion = (
                "The Sparkov-simulated dataset is nearly stationary. "
                f"Mean PSI across features and months = {mean_psi:.4f} (threshold: 0.10). "
                "This confirms the simulation produces a stable distribution with minimal "
                "natural variation. Any drift introduced in later capstone experiments is "
                "therefore injected and scripted — not naturally occurring."
            )
        elif mean_psi < 0.2:
            conclusion = (
                "The Sparkov-simulated dataset shows minor natural variation. "
                f"Mean PSI = {mean_psi:.4f} (minor shift threshold: 0.10–0.20). "
                "This is within the expected range for a simulation. "
                "Later capstone drift scenarios must still be explicitly injected and labelled."
            )
        else:
            conclusion = (
                "The Sparkov-simulated dataset shows some natural variation across months. "
                f"Mean PSI = {mean_psi:.4f} (major shift threshold: 0.20+). "
                "Report this as simulation variation, not real-world concept drift. "
                "Later drift experiments must be injected, seeded, and labelled as such."
            )
    else:
        conclusion = "Insufficient data to compute PSI conclusion."

    report = {
        "data_provenance": "synthetic/simulated (Sparkov)",
        "note": (
            "All numbers in this report come from the real Sparkov simulation. "
            "Any variation is simulation-internal, not real-world concept drift. "
            "Later capstone drift scenarios MUST be injected, seeded, config-driven, "
            "and labelled 'injected' with a known onset window."
        ),
        "summary": {
            "mean_psi_all_features": round(mean_psi, 6) if not np.isnan(mean_psi) else None,
            "val_pr_auc_mean":  round(float(val_pr.mean()),  6) if len(val_pr)  > 0 else None,
            "val_pr_auc_std":   round(float(val_pr.std()),   6) if len(val_pr)  > 1 else None,
            "test_pr_auc_mean": round(float(test_pr.mean()), 6) if len(test_pr) > 0 else None,
            "test_pr_auc_std":  round(float(test_pr.std()),  6) if len(test_pr) > 1 else None,
        },
        "conclusion": conclusion,
        "monthly_detail": monthly_rows,
    }

    report_path = ROOT_DIR / "stationarity_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    csv_path = ROOT_DIR / "stationarity_report.csv"
    report_df.to_csv(csv_path, index=False)

    logger.info("stationarity_report.json written: %d monthly records", len(monthly_rows))
    logger.info("Mean PSI: %.4f", mean_psi if not np.isnan(mean_psi) else 0)
    logger.info("Conclusion: %s", conclusion[:120])
    logger.info("Stationarity check complete.")


if __name__ == "__main__":
    main()
