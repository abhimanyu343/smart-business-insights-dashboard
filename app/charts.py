"""
Plotly chart factory for the Smartphone Market Intelligence Platform.
All charts return go.Figure objects for use in Streamlit.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Optional

BRAND_COLORS = {
    "Apple":   "#555555",
    "Samsung": "#1428A0",
    "OnePlus": "#F50514",
    "Google":  "#4285F4",
    "Xiaomi":  "#FF6900",
    "Realme":  "#FFD700",
    "Vivo":    "#415FFF",
    "Nothing": "#000000",
}

TIER_COLORS = {
    "Budget": "#2ecc71",
    "Mid-range": "#3498db",
    "Premium": "#9b59b6",
    "Ultra-premium": "#e67e22",
    "Flagship": "#e74c3c",
}


def brand_share_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of model count by brand."""
    brand_counts = df["brand"].value_counts().reset_index()
    brand_counts.columns = ["brand", "count"]
    colors = [BRAND_COLORS.get(b, "#888") for b in brand_counts["brand"]]

    fig = go.Figure(go.Pie(
        labels=brand_counts["brand"],
        values=brand_counts["count"],
        hole=0.45,
        marker_colors=colors,
        textposition="outside",
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Models: %{value}<br>Share: %{percent}<extra></extra>"
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=320
    )
    return fig


def avg_price_by_tier_bar(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: avg launch price and current price by tier."""
    tier_order = ["Budget", "Mid-range", "Premium", "Ultra-premium", "Flagship"]
    agg = df.groupby("price_tier").agg(
        avg_launch=("launch_price_inr", "mean"),
        avg_current=("current_price_inr", "mean"),
        n=("model", "count")
    ).reindex([t for t in tier_order if t in df["price_tier"].unique()])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Launch Price", x=agg.index, y=agg["avg_launch"].round(),
        marker_color=[TIER_COLORS.get(t, "#888") for t in agg.index],
        hovertemplate="<b>%{x}</b><br>Avg Launch: ₹%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="Current Price", x=agg.index, y=agg["avg_current"].round(),
        marker_color=[TIER_COLORS.get(t, "#888") for t in agg.index],
        opacity=0.6,
        hovertemplate="<b>%{x}</b><br>Avg Current: ₹%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        barmode="group", height=320,
        yaxis_title="Price (₹)",
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=20, b=20, l=40, r=20)
    )
    return fig


def price_vs_value_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: launch price vs value score, colored by brand, sized by review count."""
    fig = px.scatter(
        df, x="launch_price_inr", y="value_score",
        color="brand", size="review_count",
        hover_data=["model", "ram_gb", "main_cam_mp", "battery_mah", "user_rating"],
        color_discrete_map=BRAND_COLORS,
        opacity=0.75,
        labels={
            "launch_price_inr": "Launch Price (₹)",
            "value_score": "Value Score (0-10)",
            "brand": "Brand"
        }
    )
    # Add trendline per brand
    fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig.update_layout(
        height=460,
        xaxis_tickformat="₹,.0f",
        legend=dict(orientation="v", x=1.01),
        margin=dict(t=20, b=40, l=60, r=120)
    )
    return fig


def spec_radar_chart(df: pd.DataFrame, phone_names: List[str]) -> Optional[go.Figure]:
    """Radar chart comparing normalised specs across selected phones."""
    dims = ["ram_gb", "storage_gb", "battery_mah", "main_cam_mp",
            "refresh_rate_hz", "fast_charge_w", "value_score"]
    dim_labels = ["RAM", "Storage", "Battery", "Camera", "Refresh", "Charge Speed", "Value"]

    fig = go.Figure()
    for phone_str in phone_names:
        parts = phone_str.split(" ", 1)
        if len(parts) < 2:
            continue
        brand, model = parts[0], parts[1]
        match = df[(df["brand"] == brand) & df["model"].str.contains(model, case=False)]
        if match.empty:
            continue
        row = match.iloc[0]

        # Normalise each dimension to [0, 1] within full dataset
        vals = []
        for d in dims:
            mn, mx = df[d].min(), df[d].max()
            v = (row[d] - mn) / (mx - mn + 1e-9)
            vals.append(round(v * 10, 2))

        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=dim_labels + [dim_labels[0]],
            name=phone_str,
            fill="toself",
            opacity=0.6
        ))

    if not fig.data:
        return None

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        height=420,
        margin=dict(t=30, b=30, l=60, r=60)
    )
    return fig


def depreciation_line_chart(depr_df: pd.DataFrame) -> go.Figure:
    """Line chart showing price retention curves over 24 months."""
    months = list(range(0, 25, 3))
    fig = go.Figure()

    for _, row in depr_df.iterrows():
        lam = row["decay_lambda"]
        brand = row["brand"]
        retention = [np.exp(-lam * m * 30) * 100 for m in months]
        fig.add_trace(go.Scatter(
            x=months, y=retention,
            name=brand,
            mode="lines+markers",
            line=dict(color=BRAND_COLORS.get(brand, "#888"), width=2.5),
            hovertemplate=f"<b>{brand}</b><br>Month %{{x}}: %{{y:.1f}}% retained<extra></extra>"
        ))

    fig.update_layout(
        xaxis_title="Months since launch",
        yaxis_title="Price retained (%)",
        yaxis_ticksuffix="%",
        yaxis_range=[0, 105],
        height=380,
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=30, b=40, l=60, r=20)
    )
    fig.add_hline(y=50, line_dash="dot", line_color="gray",
                  annotation_text="50% retained", annotation_position="right")
    return fig


def feature_importance_bar(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for ML feature importance."""
    fig = px.bar(
        importance_df.head(12), x="importance" if "importance" in importance_df.columns else "mean_abs_shap",
        y="feature", orientation="h",
        color="importance" if "importance" in importance_df.columns else "mean_abs_shap",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        height=380, showlegend=False,
        xaxis_title="Importance", yaxis_title="",
        margin=dict(t=20, b=20, l=130, r=20)
    )
    return fig
