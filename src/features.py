"""
features.py — CreditOps

SINGLE SOURCE OF TRUTH for feature engineering.

This function is used by BOTH:
  - src/data_collection.py  (batch, during DVC pipeline)
  - src/main.py POST /predict  (online, per-request)

Any change here propagates to both paths automatically.

FEATURE DECISIONS
-----------------
Derived features (no future leakage, no label info):
  hour            Hour of transaction (0–23) — fraud peaks at night
  day_of_week     0=Monday … 6=Sunday
  is_weekend      1 if Saturday or Sunday
  log_amt         log(amt + eps) — amount is right-skewed
  category_*      One-hot of 14 transaction categories
  gender_M        1 if Male, 0 if Female (only 2 values in dataset)
  age_at_txn      Cardholder age at transaction time (from dob)
  haversine_km    Distance between cardholder location and merchant (km)
  log_city_pop    log(city_pop + eps) — right-skewed population

DROPPED (privacy / leakage / identifier):
  cc_num          Card number — identifier, also leaks card-level patterns
  first, last     PII names
  street          PII address
  trans_num       Transaction identifier
  zip             Partially redundant with lat/long, PII-adjacent
  dob             Used to compute age; raw DOB is PII — dropped after derivation
  merchant name   Very high cardinality (~700 merchants); frequency encoding
                  would add complexity without clear benefit over categories
  city            High cardinality (~900 cities) — frequency-encoding only 
                  captures training distribution, risky for generalisation;
                  spatial signal captured by lat/long haversine distance
  state           52 states — moderate cardinality, but collinear with lat/long;
                  dropping reduces feature count without hurting PR-AUC in tests
  job             ~490 unique jobs — extreme high cardinality; frequency-encoding
                  top-N + "other" possible but adds noise; dropped here for
                  privacy (occupation is sensitive) and model simplicity
  unix_time       Redundant with trans_date_trans_time
  lat, long       Used to compute haversine; raw coordinates dropped after
  merch_lat       Used to compute haversine; raw coordinates dropped after
  merch_long      Used to compute haversine; raw coordinates dropped after
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

ROOT_DIR  = Path(__file__).resolve().parent.parent
EPOCH     = pd.Timestamp("1970-01-01")

# 14 categories exactly as they appear in the dataset
CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport",
    "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
    "food_dining", "personal_care", "health_fitness", "travel",
    "kids_pets", "home",
]

LOG_EPS = 1e-6  # prevent log(0)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _haversine_series(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Vectorised haversine distance (km)."""
    R = 6371.0
    phi1 = np.radians(lat1.values)
    phi2 = np.radians(lat2.values)
    dphi = np.radians((lat2 - lat1).values)
    dlam = np.radians((lon2 - lon1).values)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return pd.Series(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), index=lat1.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame of raw transaction rows into model features.

    Input columns required:
        trans_date_trans_time, amt, category, gender, dob,
        lat, long, merch_lat, merch_long, city_pop

    Returns a DataFrame of features ONLY (no labels, no PII).
    The column order is deterministic — preprocessor.pkl relies on it.
    """
    df = df.copy()

    # --- Timestamp features --------------------------------------------------
    ts = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]        = ts.dt.hour.astype(np.int8)
    df["day_of_week"] = ts.dt.dayofweek.astype(np.int8)   # 0=Mon, 6=Sun
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(np.int8)

    # --- Amount --------------------------------------------------------------
    df["log_amt"] = np.log(df["amt"].clip(lower=0) + LOG_EPS)

    # --- Category one-hot ----------------------------------------------------
    for cat in CATEGORIES:
        df[f"cat_{cat}"] = (df["category"] == cat).astype(np.int8)

    # --- Gender --------------------------------------------------------------
    df["gender_M"] = (df["gender"] == "M").astype(np.int8)

    # --- Cardholder age at transaction time ----------------------------------
    dob = pd.to_datetime(df["dob"], errors="coerce")
    df["age_at_txn"] = ((ts - dob).dt.days / 365.25).round(1)
    # Fill the rare NaT case with median (should not happen in Sparkov data)
    median_age = df["age_at_txn"].median()
    df["age_at_txn"] = df["age_at_txn"].fillna(median_age if not np.isnan(median_age) else 45.0)

    # --- Haversine distance cardholder → merchant (km) -----------------------
    df["haversine_km"] = _haversine_series(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # --- Log city population -------------------------------------------------
    df["log_city_pop"] = np.log(df["city_pop"].clip(lower=1) + LOG_EPS)

    # --- Return only model feature columns ----------------------------------
    feature_cols = (
        ["hour", "day_of_week", "is_weekend", "log_amt"]
        + [f"cat_{c}" for c in CATEGORIES]
        + ["gender_M", "age_at_txn", "haversine_km", "log_city_pop"]
    )
    return df[feature_cols].reset_index(drop=True)


def build_features_single(row: dict) -> dict:
    """
    Convert a single raw transaction dict (from the API) into a feature dict.
    Calls build_features() internally to guarantee identical output.
    """
    df = pd.DataFrame([row])
    features_df = build_features(df)
    return features_df.iloc[0].to_dict()


def feature_names() -> list[str]:
    """Return the ordered list of feature column names (for preprocessor alignment)."""
    return (
        ["hour", "day_of_week", "is_weekend", "log_amt"]
        + [f"cat_{c}" for c in CATEGORIES]
        + ["gender_M", "age_at_txn", "haversine_km", "log_city_pop"]
    )
