"""
tests/fixtures.py — CreditOps test fixtures

Synthetic data generators used strictly for isolated unit testing.
Production code and offline observatory streams use real Sparkov data via DatasetAdapter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_stream_data(
    start: str = "2020-10-04",
    end: str = "2020-12-31",
    rows_per_day: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transactions for fast micro-tests.
    Used ONLY in unit tests — never in production or experiments.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    total_hours = max(1, int((end_ts - start_ts).total_seconds() // 3600))
    rng = np.random.default_rng(seed)

    n_rows = max(10, (total_hours // 24) * rows_per_day)
    times = pd.date_range(start_ts, end_ts, periods=n_rows)

    categories = [
        "misc_net", "grocery_pos", "entertainment", "gas_transport",
        "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
        "food_dining", "personal_care", "health_fitness", "travel",
        "kids_pets", "home",
    ]

    is_fraud = (rng.random(n_rows) < 0.00579).astype(int)

    return pd.DataFrame({
        "trans_date_trans_time": times,
        "amt": rng.uniform(5.0, 500.0, n_rows),
        "category": rng.choice(categories, n_rows),
        "gender": rng.choice(["M", "F"], n_rows),
        "dob": ["1980-01-01"] * n_rows,
        "lat": rng.uniform(25.0, 48.0, n_rows),
        "long": rng.uniform(-125.0, -65.0, n_rows),
        "city_pop": rng.integers(1000, 1000000, n_rows).astype(float),
        "merch_lat": rng.uniform(25.0, 48.0, n_rows),
        "merch_long": rng.uniform(-125.0, -65.0, n_rows),
        "is_fraud": is_fraud,
    })
