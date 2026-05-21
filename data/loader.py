"""
Multi-source data loader for smartphone dataset.

Priority order:
1. Local CSV (data/phones_dataset.csv)
2. Generate fresh dataset if CSV missing
3. Future: REST API / database connector
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)

DATA_PATH = Path("data/phones_dataset.csv")


def load_phones_df(path: str = None, force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load the smartphone dataset. Auto-generates if not found.

    Args:
        path: Override path to CSV file
        force_regenerate: Force regeneration even if CSV exists

    Returns:
        Cleaned, feature-enriched DataFrame
    """
    src = Path(path) if path else DATA_PATH

    if force_regenerate or not src.exists():
        log.info(f"Dataset not found at {src}. Generating...")
        from data.generate_dataset import generate_dataset
        df = generate_dataset()
        src.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(src)
        log.info(f"Generated and saved {len(df)} records to {src}")
    else:
        df = pd.read_csv(src, index_col="phone_id")
        log.info(f"Loaded {len(df)} records from {src}")

    # Apply cleaning pipeline
    from data.cleaner import SmartphoneDataCleaner
    cleaner = SmartphoneDataCleaner()
    df = cleaner.fit_transform(df)

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return key summary statistics for dashboard KPI cards."""
    return {
        "total_models":      len(df),
        "total_brands":      df["brand"].nunique(),
        "avg_launch_price":  df["launch_price_inr"].mean(),
        "median_price":      df["launch_price_inr"].median(),
        "avg_value_score":   df["value_score"].mean(),
        "pct_5g":            df["has_5g"].mean() * 100,
        "avg_rating":        df["user_rating"].mean(),
        "avg_battery":       df["battery_mah"].mean(),
        "avg_fast_charge":   df["fast_charge_w"].mean(),
        "pct_amoled":        df["display_type"].str.contains("AMOLED", case=False).mean() * 100,
    }
