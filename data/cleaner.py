"""
Data cleaning and normalisation pipeline for smartphone dataset.

Handles:
- Missing value imputation (strategy varies by column type)
- Outlier detection and capping (IQR-based)
- Type coercion and consistency checks
- Derived column validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Column schemas ─────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "ram_gb", "storage_gb", "main_cam_mp", "ultrawide_mp", "telephoto_mp",
    "front_cam_mp", "screen_size_in", "refresh_rate_hz", "battery_mah",
    "fast_charge_w", "launch_price_inr", "current_price_inr", "price_drop_pct",
    "days_since_launch", "user_rating", "review_count", "value_score", "brand_prestige"
]
BOOLEAN_COLS = ["wireless_charge", "has_5g", "nfc", "wifi6"]
CATEGORICAL_COLS = ["brand", "model", "chipset", "resolution", "display_type",
                    "usb_type", "price_tier"]
ORDINAL_COLS = {"refresh_rate_hz": [60, 90, 120, 144, 165]}


class SmartphoneDataCleaner:
    """Full cleaning pipeline for smartphone market dataset."""

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.cleaning_report: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full cleaning pipeline and return cleaned DataFrame."""
        log.info(f"Starting clean pipeline on {len(df)} records, {df.shape[1]} columns")
        df = df.copy()

        df = self._coerce_types(df)
        df = self._handle_missing(df)
        df = self._cap_outliers(df)
        df = self._validate_relationships(df)
        df = self._standardise_categoricals(df)

        log.info(f"Cleaning complete. Output: {len(df)} records")
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns have correct dtypes."""
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in BOOLEAN_COLS:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with appropriate strategies."""
        missing_before = df.isnull().sum().sum()

        # Camera MPs: 0 means absent, not missing
        for cam_col in ["ultrawide_mp", "telephoto_mp"]:
            if cam_col in df.columns:
                df[cam_col] = df[cam_col].fillna(0)

        # Numeric: median by brand (preserves brand-specific distributions)
        for col in ["ram_gb", "storage_gb", "battery_mah", "fast_charge_w",
                    "main_cam_mp", "screen_size_in", "refresh_rate_hz"]:
            if col in df.columns and df[col].isnull().any():
                df[col] = df.groupby("brand")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fallback: global median
                df[col] = df[col].fillna(df[col].median())

        # Price: if current_price missing, estimate from launch + days_since_launch
        if "current_price_inr" in df.columns:
            mask = df["current_price_inr"].isnull()
            if mask.any():
                df.loc[mask, "current_price_inr"] = (
                    df.loc[mask, "launch_price_inr"] *
                    np.exp(-0.001 * df.loc[mask, "days_since_launch"].fillna(180))
                ).round()

        # Ratings: median by price tier
        if "user_rating" in df.columns:
            df["user_rating"] = df.groupby("price_tier")["user_rating"].transform(
                lambda x: x.fillna(x.median())
            ).fillna(4.0)

        # Review count: small positive number if missing
        if "review_count" in df.columns:
            df["review_count"] = df["review_count"].fillna(100).astype(int)

        missing_after = df.isnull().sum().sum()
        self.cleaning_report["missing_imputed"] = missing_before - missing_after
        log.info(f"Imputed {missing_before - missing_after} missing values")
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap extreme outliers using IQR fencing per numeric column."""
        capped = 0
        cap_cols = ["launch_price_inr", "current_price_inr", "review_count",
                    "battery_mah", "fast_charge_w", "main_cam_mp", "days_since_launch"]

        for col in cap_cols:
            if col not in df.columns:
                continue
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.iqr_multiplier * IQR
            upper = Q3 + self.iqr_multiplier * IQR
            
            before = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=max(0, lower), upper=upper)
            capped += before

        self.cleaning_report["outliers_capped"] = capped
        log.info(f"Capped {capped} outlier values across {len(cap_cols)} columns")
        return df

    def _validate_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce logical consistency between related columns."""
        # current_price must be <= launch_price
        if all(c in df.columns for c in ["current_price_inr", "launch_price_inr"]):
            mask = df["current_price_inr"] > df["launch_price_inr"]
            df.loc[mask, "current_price_inr"] = df.loc[mask, "launch_price_inr"]

        # price_drop_pct recalculated from actual prices
        if all(c in df.columns for c in ["launch_price_inr", "current_price_inr"]):
            df["price_drop_pct"] = (
                (1 - df["current_price_inr"] / df["launch_price_inr"]) * 100
            ).clip(lower=0).round(1)

        # user_rating must be [1, 5]
        if "user_rating" in df.columns:
            df["user_rating"] = df["user_rating"].clip(1.0, 5.0)

        # refresh_rate should be in valid set
        if "refresh_rate_hz" in df.columns:
            valid_rates = {60, 90, 120, 144, 165}
            invalid = ~df["refresh_rate_hz"].isin(valid_rates)
            df.loc[invalid, "refresh_rate_hz"] = df.loc[invalid, "refresh_rate_hz"].apply(
                lambda x: min(valid_rates, key=lambda v: abs(v - x))
            )

        return df

    def _standardise_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Title-case brands, strip whitespace, normalise display types."""
        if "brand" in df.columns:
            df["brand"] = df["brand"].str.strip().str.title()
        if "display_type" in df.columns:
            # Normalise AMOLED variants
            df["display_type"] = df["display_type"].str.replace(
                r"(?i)super amoled", "Super AMOLED", regex=True
            )
        return df

    def get_report(self) -> dict:
        """Return cleaning summary report."""
        return self.cleaning_report


def clean_dataset(path: str = "data/phones_dataset.csv") -> pd.DataFrame:
    """Convenience wrapper — load and clean in one call."""
    df = pd.read_csv(path, index_col="phone_id")
    cleaner = SmartphoneDataCleaner()
    return cleaner.fit_transform(df)
