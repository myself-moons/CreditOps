"""
src/observatory/drift/injector.py — CreditOps v2 Observatory, Phase 1

Seeded drift injector for offline stream simulation.

Takes a window of V1 FEATURE data (already built by build_features from
src/features.py — do NOT duplicate feature logic here) and applies the
configured drift to windows at or after onset_window.

IMPORTANT:
  - All randomness is seeded via numpy.random.default_rng(seed + window_index)
    so each window is reproducible given the same global seed.
  - Raw transaction columns (amt, lat, long, etc.) are NOT required here;
    the injector operates primarily on the feature DataFrame returned by build_features().
  - The label column (is_fraud) is passed in separately and returned modified
    for fraud_rate_shift, concept_drift, and novel_pattern scenarios.
  - For covariate_shift, only feature values are shifted; labels are untouched.
  - For control, nothing is changed.

Scenario definitions:
  control          — no change (used to measure false alarms).

  covariate_shift  — shift distribution of transaction amount (log_amt) and
                     cardholder-merchant distance (haversine_km) upward by `magnitude`,
                     labels untouched.

  fraud_rate_shift — resample/relabel so the fraud rate in affected windows rises to
                     `magnitude` x baseline (baseline 0.579%, from training set) by
                     oversampling real fraud rows. Do not fabricate labels on legitimate rows.

  concept_drift    — change the feature-to-fraud relationship while keeping the
                     marginal feature distributions similar, e.g. swap which merchant
                     categories carry the fraud, by moving fraud labels to rows of
                     different categories. Documented in docs/observatory_data_dictionary.md.

  novel_pattern    — introduce a fraud typology unseen in training: fraud rows get a
                     new merchant category value not present in the training set
                     (all cat_* set to 0 in one-hot features) and a distinctive
                     amount/time/distance signature. Preprocessing handles unseen
                     categories without crashing.

Gradual shape linearly ramps the effect from 0 to full over ramp_windows.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sparkov training baseline fraud rate (0.579% as recorded in split_info.json)
BASE_FRAUD_RATE = 0.005789

# Feature names from V1
_LOG_AMT_COL = "log_amt"
_HAVERSINE_COL = "haversine_km"
_IS_WEEKEND_COL = "is_weekend"
_HOUR_COL = "hour"
_CAT_PREFIX = "cat_"


def _effective_magnitude(
    window_index: int,
    onset_window: int,
    ramp_windows: int,
    magnitude: float,
    shape: str,
) -> float:
    """
    Compute the effective magnitude at this window_index given shape.

    sudden:  full magnitude from onset_window onward, 0 before.
    gradual: linearly ramps from 0 at onset_window to full magnitude
             at onset_window + ramp_windows - 1.
    """
    if window_index < onset_window:
        return 0.0
    if shape == "sudden":
        return float(magnitude)
    # gradual
    steps_in = window_index - onset_window
    ramp_frac = min(steps_in / max(ramp_windows - 1, 1), 1.0)
    return float(magnitude * ramp_frac)


def apply_drift(
    features_df: pd.DataFrame,
    labels: pd.Series,
    window_index: int,
    *,
    scenario: str,
    shape: str,
    onset_window: int,
    ramp_windows: int,
    magnitude: float,
    affected_categories: Optional[Sequence[str]] = None,
    seed: int = 42,
    base_fraud_rate: float = BASE_FRAUD_RATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply the configured drift scenario to one stream window.

    Args:
        features_df:        Output of build_features() — feature columns only.
        labels:             Corresponding is_fraud Series (same index as features_df).
        window_index:       0-indexed position of this window in the stream.
        scenario:           One of: control, covariate_shift, fraud_rate_shift,
                            concept_drift, novel_pattern.
        shape:              'sudden' or 'gradual'.
        onset_window:       Window index at which drift begins.
        ramp_windows:       Number of windows to ramp up (gradual only).
        magnitude:          Scenario-specific float.
        affected_categories: Optional list of category names (e.g. ["travel"]).
        seed:               Global seed; per-window RNG is seeded with seed + window_index.
        base_fraud_rate:    Baseline fraud rate to scale in fraud_rate_shift (default 0.005789).

    Returns:
        (modified_features_df, modified_labels) — copies, originals untouched.
    """
    # Validation: empty window handling
    if features_df.empty:
        return features_df.copy(), labels.copy()

    # Pre-drift or control: no modification
    eff_magnitude = _effective_magnitude(
        window_index, onset_window, ramp_windows, magnitude, shape
    )
    if eff_magnitude == 0.0 or scenario == "control":
        return features_df.copy(), labels.copy()

    # Per-window reproducible RNG — guarantees same_seed_same_result
    rng = np.random.default_rng(seed + window_index)
    feats = features_df.copy().reset_index(drop=True)
    lbls = labels.copy().reset_index(drop=True)

    if scenario == "covariate_shift":
        feats = _apply_covariate_shift(feats, eff_magnitude, rng)

    elif scenario == "fraud_rate_shift":
        feats, lbls = _apply_fraud_rate_shift(feats, lbls, eff_magnitude, rng, base_fraud_rate=base_fraud_rate)

    elif scenario == "concept_drift":
        feats, lbls = _apply_concept_drift(
            feats, lbls, eff_magnitude, list(affected_categories or []), rng
        )

    elif scenario == "novel_pattern":
        feats, lbls = _apply_novel_pattern(feats, lbls, eff_magnitude, rng)

    else:
        logger.warning("Unknown scenario '%s' — treating as control.", scenario)

    return feats, lbls


# ──────────────────────────────────────────────────────────────────────────── #
# Scenario implementations                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

def _apply_covariate_shift(
    feats: pd.DataFrame,
    magnitude: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Shift distribution of transaction amount (log_amt) and
    cardholder-merchant distance (haversine_km) upward by `magnitude`.
    Labels remain untouched.
    """
    for col in [_LOG_AMT_COL, _HAVERSINE_COL]:
        if col not in feats.columns:
            raise ValueError(f"features_df is missing required column: '{col}' for covariate_shift")

    n = len(feats)
    noise = rng.normal(0, 0.05, size=n)

    # Shift log_amt upward
    feats[_LOG_AMT_COL] = feats[_LOG_AMT_COL] + magnitude + noise

    # Shift haversine_km upward by magnitude relative to typical scale
    med_hav = feats[_HAVERSINE_COL].median()
    scale = med_hav if (not np.isnan(med_hav) and med_hav > 0) else 50.0
    feats[_HAVERSINE_COL] = (
        feats[_HAVERSINE_COL] + (magnitude * scale) + (noise * scale * 0.1)
    ).clip(lower=0.0)

    logger.debug(
        "covariate_shift applied: log_amt +%.3f, haversine_km +%.3f",
        magnitude, magnitude * scale,
    )
    return feats


def _apply_fraud_rate_shift(
    feats: pd.DataFrame,
    labels: pd.Series,
    magnitude: float,
    rng: np.random.Generator,
    base_fraud_rate: float = BASE_FRAUD_RATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Resample/relabel so the fraud rate in affected windows rises to
    `magnitude` x baseline (baseline 0.579%) by oversampling real fraud rows.
    Do not fabricate labels on legitimate rows.
    """
    fraud_idx = labels[labels == 1].index.to_numpy()
    legit_idx = labels[labels == 0].index.to_numpy()

    if len(fraud_idx) == 0:
        logger.warning("fraud_rate_shift: window has no real fraud rows to oversample. Returning unchanged.")
        return feats, labels

    current_rate = float(len(fraud_idx) / len(labels))
    target_rate = min(float(magnitude * max(current_rate, base_fraud_rate)), 0.95)
    n_legit = len(legit_idx)

    # Calculate desired fraud count to hit target_rate: N_fraud / (N_fraud + N_legit) = target_rate
    # N_fraud = target_rate * N_legit / (1 - target_rate)
    target_fraud_count = int(round((target_rate / max(1.0 - target_rate, 1e-6)) * n_legit))
    current_fraud_count = len(fraud_idx)

    if target_fraud_count > current_fraud_count:
        # Oversample real fraud rows (sample with replacement)
        n_extra = target_fraud_count - current_fraud_count
        sampled_fraud_indices = rng.choice(fraud_idx, size=n_extra, replace=True)

        extra_feats = feats.iloc[sampled_fraud_indices].copy()
        extra_labels = labels.iloc[sampled_fraud_indices].copy()

        feats = pd.concat([feats, extra_feats], ignore_index=True)
        labels = pd.concat([labels, extra_labels], ignore_index=True)

        logger.debug(
            "fraud_rate_shift: oversampled %d real fraud rows. Rate: %.4f (target: %.4f)",
            n_extra, labels.mean(), target_rate,
        )
    elif target_fraud_count < current_fraud_count and target_fraud_count > 0:
        # Downsample real fraud rows
        kept_fraud_indices = rng.choice(fraud_idx, size=target_fraud_count, replace=False)
        all_kept_indices = np.concatenate([legit_idx, kept_fraud_indices])
        all_kept_indices.sort()

        feats = feats.iloc[all_kept_indices].reset_index(drop=True)
        labels = labels.iloc[all_kept_indices].reset_index(drop=True)

        logger.debug(
            "fraud_rate_shift: downsampled real fraud rows to %d. Rate: %.4f (target: %.4f)",
            target_fraud_count, labels.mean(), target_rate,
        )

    return feats, labels


def _apply_concept_drift(
    feats: pd.DataFrame,
    labels: pd.Series,
    magnitude: float,
    affected_categories: list,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Change the feature-to-fraud relationship while keeping the
    marginal feature distributions identical:
    Post-onset fraud concentrates in a fixed small set of merchant categories
    plus a fixed amount band and hour range (a learnable pattern different from
    the training pattern), keeping fraud count and marginal features roughly similar.
    """
    fraud_idx = labels[labels == 1].index.to_numpy()
    legit_idx = labels[labels == 0].index.to_numpy()

    if len(fraud_idx) == 0 or len(legit_idx) == 0:
        logger.debug("concept_drift: window lacks sufficient fraud or legit rows to swap.")
        return feats, labels

    flip_frac = float(np.clip(magnitude, 0.0, 1.0))
    n_swap = max(1, int(round(flip_frac * len(fraud_idx))))
    n_swap = min(n_swap, len(fraud_idx), len(legit_idx))

    # Pick fraud rows to demote
    chosen_fraud = rng.choice(fraud_idx, size=n_swap, replace=False)

    # Determine candidate legitimate rows to promote to fraud
    # Concentrate in fixed small set of merchant categories plus fixed amount band/hour range
    target_cats = list(affected_categories) if affected_categories else ["gas_transport", "grocery_net"]
    cat_cols = [c for c in feats.columns if c.startswith(_CAT_PREFIX)]

    target_pool = None
    if cat_cols:
        matched_cols = [f"{_CAT_PREFIX}{c}" for c in target_cats if f"{_CAT_PREFIX}{c}" in feats.columns]
        if matched_cols:
            cat_mask = (feats.iloc[legit_idx][matched_cols].sum(axis=1) > 0).to_numpy()

            # Amount band ($35 to $180, i.e., log_amt in [3.5, 5.2]) and daytime hours (11 to 17)
            hr_mask = np.ones(len(legit_idx), dtype=bool)
            amt_mask = np.ones(len(legit_idx), dtype=bool)
            if "hour" in feats.columns:
                hr_val = feats.iloc[legit_idx]["hour"].to_numpy()
                hr_mask = (hr_val >= 11) & (hr_val <= 17)
            if _LOG_AMT_COL in feats.columns:
                amt_val = feats.iloc[legit_idx][_LOG_AMT_COL].to_numpy()
                amt_mask = (amt_val >= 3.5) & (amt_val <= 5.2)

            strict_match = cat_mask & hr_mask & amt_mask
            if strict_match.sum() >= n_swap:
                target_pool = legit_idx[strict_match]
            elif cat_mask.sum() >= n_swap:
                target_pool = legit_idx[cat_mask]

    if target_pool is None or len(target_pool) < n_swap:
        target_pool = legit_idx

    chosen_legit = rng.choice(target_pool, size=n_swap, replace=False)

    lbls = labels.copy()
    lbls.iloc[chosen_fraud] = 0
    lbls.iloc[chosen_legit] = 1

    logger.debug(
        "concept_drift: swapped %d fraud labels to rows in %s (flip_frac=%.3f)",
        n_swap, target_cats, flip_frac,
    )
    return feats, lbls


def _apply_novel_pattern(
    feats: pd.DataFrame,
    labels: pd.Series,
    magnitude: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Introduce a fraud typology unseen in training:
    Fraud rows get a new merchant category value not present in the training set
    (all 14 cat_* one-hot columns set to 0) and a distinctive amount/time/distance signature.
    Preprocessing (ColumnTransformer) does not crash on this unseen category representation.
    """
    fraud_idx = labels[labels == 1].index.to_numpy()
    if len(fraud_idx) == 0:
        logger.debug("novel_pattern: no fraud rows in window to apply novel pattern.")
        return feats, labels

    replace_frac = float(np.clip(magnitude, 0.0, 1.0))
    n_replace = max(1, int(round(replace_frac * len(fraud_idx))))
    chosen_fraud = rng.choice(fraud_idx, size=min(n_replace, len(fraud_idx)), replace=False)

    feats = feats.copy()

    # 1. Unseen merchant category: zero out all known one-hot category columns
    cat_cols = [c for c in feats.columns if c.startswith(_CAT_PREFIX)]
    for c in cat_cols:
        feats.loc[chosen_fraud, c] = 0

    # 2. Distinctive high amount signature
    if _LOG_AMT_COL in feats.columns:
        p95_amt = float(np.nanpercentile(feats[_LOG_AMT_COL].values, 95))
        feats.loc[chosen_fraud, _LOG_AMT_COL] = p95_amt + (2.0 * max(magnitude, 0.5))

    # 3. Distinctive long distance signature
    if _HAVERSINE_COL in feats.columns:
        p95_hav = float(np.nanpercentile(feats[_HAVERSINE_COL].values, 95))
        feats.loc[chosen_fraud, _HAVERSINE_COL] = p95_hav + (100.0 * max(magnitude, 0.5))

    # 4. Distinctive timing signature: weekend night
    if _IS_WEEKEND_COL in feats.columns:
        feats.loc[chosen_fraud, _IS_WEEKEND_COL] = 1
    if _HOUR_COL in feats.columns:
        feats.loc[chosen_fraud, _HOUR_COL] = 3

    logger.debug(
        "novel_pattern: applied novel typology to %d/%d fraud rows",
        len(chosen_fraud), len(fraud_idx),
    )
    return feats, labels
