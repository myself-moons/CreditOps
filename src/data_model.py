"""
data_model.py — CreditOps

Pydantic request model for POST /predict.
Fields match exactly what src/features.build_features() expects.

⚠ SIMULATED DATA NOTICE: This API serves a model trained on Sparkov-simulated
credit-card transactions. No real cardholder data is used.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator

# The 14 category values from the Sparkov dataset
CATEGORIES = Literal[
    "misc_net", "grocery_pos", "entertainment", "gas_transport",
    "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
    "food_dining", "personal_care", "health_fitness", "travel",
    "kids_pets", "home",
]


class Transaction(BaseModel):
    """
    Raw transaction fields required to compute model features.
    Pydantic validates ranges and enum values before any model code runs.

    All fields map to the Sparkov simulation schema.
    PII fields (cc_num, first, last, street, trans_num) are intentionally absent —
    they are dropped by the feature engineering step.
    """

    # Timestamp (ISO 8601 or parseable string)
    trans_date_trans_time: str = Field(
        ...,
        description="Transaction timestamp, e.g. '2020-06-15 14:32:00'",
        examples=["2020-06-15 14:32:00"],
    )

    # Transaction details
    category: CATEGORIES = Field(..., description="Merchant category (14 values)")
    amt: float = Field(..., ge=0.0, description="Transaction amount in USD")

    # Cardholder demographics (no PII names or card numbers)
    gender: Literal["M", "F"] = Field(..., description="Cardholder gender (M or F)")
    dob: str = Field(..., description="Date of birth, e.g. '1980-03-25'", examples=["1980-03-25"])

    # Location
    lat:       float = Field(..., ge=-90.0,  le=90.0,  description="Cardholder latitude")
    long:      float = Field(..., ge=-180.0, le=180.0, description="Cardholder longitude")
    city_pop:  int   = Field(..., ge=0,                description="Population of cardholder city")
    merch_lat:  float = Field(..., ge=-90.0,  le=90.0,  description="Merchant latitude")
    merch_long: float = Field(..., ge=-180.0, le=180.0, description="Merchant longitude")

    @field_validator("trans_date_trans_time")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        import pandas as pd
        try:
            ts = pd.to_datetime(v)
            if ts.year < 1970 or ts.year > 2050:
                raise ValueError(f"Year {ts.year} is out of acceptable bounds (1970 to 2050).")
        except Exception as e:
            if isinstance(e, ValueError) and "acceptable bounds" in str(e):
                raise
            raise ValueError(f"Invalid transaction date/time '{v}'. Expected format like 'YYYY-MM-DD HH:MM:SS'.") from e
        return v

    @field_validator("dob")
    @classmethod
    def validate_dob(cls, v: str) -> str:
        import pandas as pd
        try:
            ts = pd.to_datetime(v)
            if ts.year < 1900 or ts.year > 2025:
                raise ValueError(f"Birth year {ts.year} is out of realistic cardholder bounds (1900 to 2025).")
        except Exception as e:
            if isinstance(e, ValueError) and "realistic cardholder bounds" in str(e):
                raise
            raise ValueError(f"Invalid date of birth '{v}'. Expected format like 'YYYY-MM-DD'.") from e
        return v

    model_config = {"populate_by_name": True}