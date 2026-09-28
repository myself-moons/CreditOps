"""
data_preprocessing.py — CreditOps

DVC Stage: Data_Preprocessing
------------------------------
Loads data/raw/{train,val,test}.csv, calls src/features.build_features()
on each split, fits an sklearn ColumnTransformer on TRAIN ONLY,
transforms all three splits, saves preprocessed CSVs and preprocessor.pkl.

IMPORTANT: The preprocessor is fit on training data only.
           No validation or test data touches the fitting step.

SIMULATED DATA NOTICE: Source data is Sparkov simulation.
"""

import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import build_features, feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "is_fraud"

# ── Column groups (aligned with feature_names() output) ──────────────────── #
# All features from build_features() are either binary (0/1) or continuous.
# Binary / one-hot features need no scaling; continuous features do.
CONTINUOUS_FEATURES = [
    "hour", "day_of_week",         # ordinal but treated as continuous
    "log_amt", "age_at_txn",
    "haversine_km", "log_city_pop",
]
# Everything else (is_weekend, cat_*, gender_M) is already in {0, 1}
# and passes through unchanged.


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales continuous features.
    Binary/one-hot features pass through unchanged.
    Fitted on training data only.
    """
    all_features = feature_names()
    passthrough   = [f for f in all_features if f not in CONTINUOUS_FEATURES]

    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), CONTINUOUS_FEATURES),
            ("passthrough", "passthrough", passthrough),
        ],
        remainder="drop",
    )


def main() -> None:
    raw_dir = ROOT_DIR / "data" / "raw"
    out_dir = ROOT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== CreditOps Data Preprocessing ===")
    logger.info("NOTICE: Source data is simulated (Sparkov). No real cardholders involved.")

    for split in ("train", "val", "test"):
        path = raw_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run data_collection.py first (dvc repro Data_Collection)."
            )

    train = pd.read_csv(raw_dir / "train.csv", low_memory=False)
    val   = pd.read_csv(raw_dir / "val.csv",   low_memory=False)
    test  = pd.read_csv(raw_dir / "test.csv",  low_memory=False)

    logger.info("Train %d rows | Val %d rows | Test %d rows",
                len(train), len(val), len(test))

    # Separate labels
    y_train = train[TARGET].values
    y_val   = val[TARGET].values
    y_test  = test[TARGET].values

    # Build feature matrices via the single source of truth
    logger.info("Building features (src/features.py)...")
    X_train = build_features(train)
    X_val   = build_features(val)
    X_test  = build_features(test)

    logger.info("Feature shape: %s", X_train.shape)

    # Fit preprocessor on TRAIN ONLY
    logger.info("Fitting preprocessor on training data only...")
    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)
    X_test_enc  = preprocessor.transform(X_test)

    # Recover column names
    cont_names = CONTINUOUS_FEATURES
    pass_names = [f for f in feature_names() if f not in CONTINUOUS_FEATURES]
    col_names  = cont_names + pass_names

    def save_split(X_enc, y, name: str) -> None:
        df = pd.DataFrame(X_enc, columns=col_names)
        df[TARGET] = y
        path = out_dir / f"{name}_processed.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %s (%d rows, %d features)", path.name, len(df), len(col_names))

    save_split(X_train_enc, y_train, "train")
    save_split(X_val_enc,   y_val,   "val")
    save_split(X_test_enc,  y_test,  "test")

    # Save fitted preprocessor
    preprocessor_path = ROOT_DIR / "preprocessor.pkl"
    with preprocessor_path.open("wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("preprocessor.pkl saved to %s", preprocessor_path)

    logger.info("Data preprocessing complete.")


if __name__ == "__main__":
    main()