"""
Cross-brand and cross-model benchmarking engine.

Computes:
- Brand scorecards across 8 dimensions
- Per-spec value density (spec per ₹1000)
- Head-to-head phone comparisons
- Price tier champion identification
- Competitive positioning matrix
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


# ── Dimension weights for overall brand score ─────────────────────────────────
SCORECARD_WEIGHTS = {
    "camera_score":      0.20,
    "performance_score": 0.20,
    "battery_score":     0.15,
    "display_score":     0.15,
    "value_score":       0.20,
    "build_score":       0.10,
}


def compute_brand_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a multi-dimension scorecard for each brand.
    Scores are normalised 0-10 within the dataset.

    Returns DataFrame indexed by brand with 8 score columns + overall.
    """
    def norm(series: pd.Series) -> pd.Series:
        """Min-max normalise to 0-10."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([5.0] * len(series), index=series.index)
        return (series - mn) / (mx - mn) * 10

    g = df.groupby("brand").agg(
        avg_main_cam=("main_cam_mp", "mean"),
        pct_has_ultra=("ultrawide_mp", lambda x: (x > 0).mean()),
        pct_has_tele=("telephoto_mp", lambda x: (x > 0).mean()),
        avg_refresh=("refresh_rate_hz", "mean"),
        pct_amoled=("display_type", lambda x: x.str.contains("AMOLED", case=False).mean()),
        avg_battery=("battery_mah", "mean"),
        avg_charge=("fast_charge_w", "mean"),
        pct_wireless=("wireless_charge", "mean"),
        avg_ram=("ram_gb", "mean"),
        avg_storage=("storage_gb", "mean"),
        avg_rating=("user_rating", "mean"),
        avg_review_count=("review_count", "mean"),
        avg_value_score=("value_score", "mean"),
        avg_price=("launch_price_inr", "mean"),
        pct_5g=("has_5g", "mean"),
        pct_nfc=("nfc", "mean"),
        model_count=("model", "nunique"),
    )

    # Camera score: weighted sum of sensors + resolution
    g["camera_score"] = norm(
        g["avg_main_cam"] * 0.5 +
        g["pct_has_ultra"] * 20 +
        g["pct_has_tele"] * 15
    )

    # Performance score: RAM + storage + 5G
    g["performance_score"] = norm(
        g["avg_ram"] * 0.8 +
        np.log1p(g["avg_storage"]) * 1.5 +
        g["pct_5g"] * 5
    )

    # Battery score
    g["battery_score"] = norm(
        g["avg_battery"] / 500 +
        g["avg_charge"] * 0.1 +
        g["pct_wireless"] * 2
    )

    # Display score
    g["display_score"] = norm(
        g["avg_refresh"] / 15 +
        g["pct_amoled"] * 5
    )

    # Value score (from pre-computed column)
    g["value_score_norm"] = norm(g["avg_value_score"])

    # Build quality proxy: NFC + wireless charge + brand prestige (avg across models)
    prestige = df.groupby("brand")["brand_prestige"].first()
    g["build_score"] = norm(g["pct_nfc"] * 3 + g["pct_wireless"] * 2 + prestige)

    # Overall weighted score
    g["overall_score"] = (
        g["camera_score"]       * SCORECARD_WEIGHTS["camera_score"] +
        g["performance_score"]  * SCORECARD_WEIGHTS["performance_score"] +
        g["battery_score"]      * SCORECARD_WEIGHTS["battery_score"] +
        g["display_score"]      * SCORECARD_WEIGHTS["display_score"] +
        g["value_score_norm"]   * SCORECARD_WEIGHTS["value_score"] +
        g["build_score"]        * SCORECARD_WEIGHTS["build_score"]
    ).round(2)

    scorecard_cols = ["camera_score", "performance_score", "battery_score",
                      "display_score", "value_score_norm", "build_score", "overall_score",
                      "avg_price", "avg_rating", "model_count"]

    return g[scorecard_cols].rename(columns={"value_score_norm": "value_score"}).sort_values(
        "overall_score", ascending=False
    ).round(2)


def head_to_head(df: pd.DataFrame, phone_a: str, phone_b: str,
                 brand_a: Optional[str] = None, brand_b: Optional[str] = None) -> pd.DataFrame:
    """
    Side-by-side comparison of two specific phones.
    Returns long-format DataFrame suitable for Plotly radar/bar charts.

    Args:
        df: Full dataset
        phone_a / phone_b: Model names (partial match supported)
        brand_a / brand_b: Optional brand filter to disambiguate

    Returns:
        DataFrame with columns [spec, phone_a_value, phone_b_value, winner]
    """
    def find_phone(model_query: str, brand_filter: Optional[str]) -> pd.Series:
        mask = df["model"].str.contains(model_query, case=False, na=False)
        if brand_filter:
            mask &= df["brand"].str.contains(brand_filter, case=False, na=False)
        candidates = df[mask]
        if candidates.empty:
            raise ValueError(f"Phone not found: '{model_query}' (brand={brand_filter})")
        # Return median row to handle multiple variants
        return candidates.select_dtypes("number").median()

    row_a = find_phone(phone_a, brand_a)
    row_b = find_phone(phone_b, brand_b)

    specs = [
        ("RAM (GB)", "ram_gb"),
        ("Storage (GB)", "storage_gb"),
        ("Main Camera (MP)", "main_cam_mp"),
        ("Ultrawide (MP)", "ultrawide_mp"),
        ("Battery (mAh)", "battery_mah"),
        ("Fast Charge (W)", "fast_charge_w"),
        ("Screen Size (in)", "screen_size_in"),
        ("Refresh Rate (Hz)", "refresh_rate_hz"),
        ("Launch Price (₹)", "launch_price_inr"),
        ("Current Price (₹)", "current_price_inr"),
        ("User Rating", "user_rating"),
        ("Value Score", "value_score"),
    ]

    rows = []
    for label, col in specs:
        if col not in row_a.index:
            continue
        val_a = row_a[col]
        val_b = row_b[col]
        # Winner: higher is better except price (lower is better)
        if col in ["launch_price_inr", "current_price_inr"]:
            winner = phone_a if val_a < val_b else (phone_b if val_b < val_a else "Tie")
        else:
            winner = phone_a if val_a > val_b else (phone_b if val_b > val_a else "Tie")
        rows.append({
            "spec": label,
            f"{phone_a}": round(val_a, 1),
            f"{phone_b}": round(val_b, 1),
            "winner": winner
        })

    return pd.DataFrame(rows)


def price_tier_champions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each price tier, identify the top 3 value champions.
    Ranking by value_score, then user_rating, then review_count.
    """
    tier_order = ["Budget", "Mid-range", "Premium", "Ultra-premium", "Flagship"]
    results = []

    for tier in tier_order:
        subset = df[df["price_tier"] == tier].copy()
        if subset.empty:
            continue
        top3 = subset.nlargest(3, ["value_score", "user_rating", "review_count"])
        for rank, (_, row) in enumerate(top3.iterrows(), 1):
            results.append({
                "price_tier": tier,
                "rank": rank,
                "brand": row["brand"],
                "model": row["model"],
                "launch_price_inr": row["launch_price_inr"],
                "value_score": row["value_score"],
                "user_rating": row["user_rating"],
                "review_count": int(row["review_count"]),
                "ram_gb": row["ram_gb"],
                "main_cam_mp": row["main_cam_mp"],
                "battery_mah": row["battery_mah"],
            })

    return pd.DataFrame(results).set_index(["price_tier", "rank"])


def spec_per_rupee(df: pd.DataFrame, price_col: str = "launch_price_inr") -> pd.DataFrame:
    """
    Compute spec density: how much of each spec you get per ₹10,000.
    Useful for identifying which brands give the most value.
    """
    price_unit = 10_000
    result = df.groupby("brand").apply(lambda g: pd.Series({
        "ram_per_10k":     (g["ram_gb"] / g[price_col] * price_unit).median().round(3),
        "storage_per_10k": (g["storage_gb"] / g[price_col] * price_unit).median().round(1),
        "battery_per_10k": (g["battery_mah"] / g[price_col] * price_unit).median().round(0),
        "charge_per_10k":  (g["fast_charge_w"] / g[price_col] * price_unit).median().round(2),
        "camera_per_10k":  (g["main_cam_mp"] / g[price_col] * price_unit).median().round(2),
        "avg_price_inr":   g[price_col].median().round(0),
        "n_models":        g["model"].nunique()
    }), include_groups=False)

    return result.sort_values("ram_per_10k", ascending=False)
