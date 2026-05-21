"""
Smartphone Market Intelligence Platform — Streamlit entry point.

Multi-page app with sidebar navigation. Each page is a self-contained module
in app/pages/ that imports from analytics/ and data/.

Run: streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_phones_df
from app.charts import (
    brand_share_donut, avg_price_by_tier_bar, spec_radar_chart,
    price_vs_value_scatter, depreciation_line_chart, feature_importance_bar
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📱 Smartphone Market Intelligence",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1f77b4;
    }
    .winner-badge {
        background: #d4edda;
        color: #155724;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .insight-box {
        background: #e8f4fd;
        border-left: 4px solid #17a2b8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_phones_df()


def render_sidebar(df: pd.DataFrame) -> dict:
    """Render sidebar filters and return active filter state."""
    st.sidebar.title("🔍 Filters")

    brands = st.sidebar.multiselect(
        "Brand", sorted(df["brand"].unique()),
        default=sorted(df["brand"].unique())
    )
    price_min, price_max = int(df["launch_price_inr"].min()), int(df["launch_price_inr"].max())
    price_range = st.sidebar.slider(
        "Launch Price (₹)", price_min, price_max, (price_min, price_max),
        step=1000, format="₹%d"
    )
    tiers = st.sidebar.multiselect(
        "Price Tier", df["price_tier"].unique().tolist(),
        default=df["price_tier"].unique().tolist()
    )
    years = st.sidebar.multiselect(
        "Release Year", sorted(df["release_year"].unique(), reverse=True),
        default=sorted(df["release_year"].unique(), reverse=True)
    )
    only_5g = st.sidebar.checkbox("5G Only", value=False)

    return {
        "brands": brands,
        "price_range": price_range,
        "tiers": tiers,
        "years": years,
        "only_5g": only_5g
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = (
        df["brand"].isin(filters["brands"]) &
        df["launch_price_inr"].between(*filters["price_range"]) &
        df["price_tier"].isin(filters["tiers"]) &
        df["release_year"].isin(filters["years"])
    )
    if filters["only_5g"]:
        mask &= df["has_5g"]
    return df[mask]


def page_market_overview(df: pd.DataFrame) -> None:
    """Page 1: Market overview with KPI cards and distribution charts."""
    st.title("📊 Market Overview")
    st.caption(f"Analysing {len(df):,} smartphone models across {df['brand'].nunique()} brands")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Models", f"{len(df):,}")
    c2.metric("Avg Launch Price", f"₹{df['launch_price_inr'].mean():,.0f}")
    c3.metric("Avg Value Score", f"{df['value_score'].mean():.1f}/10")
    c4.metric("5G Penetration", f"{df['has_5g'].mean()*100:.0f}%")
    c5.metric("Avg User Rating", f"{df['user_rating'].mean():.2f} ⭐")

    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("📦 Model Distribution by Brand")
        fig = brand_share_donut(df)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("💰 Average Price by Tier")
        fig = avg_price_by_tier_bar(df)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("💡 Price vs Value — Where is each brand positioned?")
    fig = price_vs_value_scatter(df)
    st.plotly_chart(fig, use_container_width=True)

    # Insight boxes
    top_value = df.nlargest(1, "value_score").iloc[0]
    most_reviewed = df.nlargest(1, "review_count").iloc[0]

    st.markdown(f"""
    <div class="insight-box">💡 <b>Best value phone in current selection:</b> {top_value["brand"]} {top_value["model"]}
    — Value Score {top_value["value_score"]:.1f}/10 at ₹{top_value["launch_price_inr"]:,.0f}</div>
    <div class="insight-box">📣 <b>Most reviewed:</b> {most_reviewed["brand"]} {most_reviewed["model"]}
    with {most_reviewed["review_count"]:,} reviews (Rating: {most_reviewed["user_rating"]} ⭐)</div>
    """, unsafe_allow_html=True)


def page_spec_benchmarker(df: pd.DataFrame) -> None:
    """Page 2: Select and compare up to 4 phones side by side."""
    st.title("⚡ Spec Benchmarker")

    all_phones = (df["brand"] + " " + df["model"]).unique().tolist()
    selected = st.multiselect(
        "Select phones to compare (2–4)",
        sorted(all_phones), max_selections=4,
        default=sorted(all_phones)[:2]
    )

    if len(selected) < 2:
        st.info("Select at least 2 phones to compare.")
        return

    specs_to_compare = [
        "ram_gb", "storage_gb", "main_cam_mp", "battery_mah",
        "fast_charge_w", "refresh_rate_hz", "user_rating", "value_score",
        "launch_price_inr", "current_price_inr", "price_drop_pct"
    ]
    labels = {
        "ram_gb": "RAM (GB)", "storage_gb": "Storage (GB)",
        "main_cam_mp": "Main Camera (MP)", "battery_mah": "Battery (mAh)",
        "fast_charge_w": "Fast Charge (W)", "refresh_rate_hz": "Refresh Rate (Hz)",
        "user_rating": "User Rating", "value_score": "Value Score",
        "launch_price_inr": "Launch Price (₹)", "current_price_inr": "Current Price (₹)",
        "price_drop_pct": "Price Drop (%)"
    }
    lower_better = {"launch_price_inr", "current_price_inr", "price_drop_pct"}

    rows = []
    phone_data = {}
    for phone_str in selected:
        brand, *model_parts = phone_str.split(" ")
        model = " ".join(model_parts)
        match = df[(df["brand"] == brand) & (df["model"].str.contains(model, case=False))]
        if not match.empty:
            phone_data[phone_str] = match.iloc[0]

    for spec in specs_to_compare:
        vals = {p: phone_data[p][spec] for p in selected if p in phone_data}
        best = min(vals, key=vals.get) if spec in lower_better else max(vals, key=vals.get)
        row = {"Spec": labels.get(spec, spec)}
        for p in selected:
            v = vals.get(p, "N/A")
            row[p] = f"{'✅ ' if p == best else ''}{v:,.0f}" if isinstance(v, (int, float)) else str(v)
        rows.append(row)

    compare_df = pd.DataFrame(rows).set_index("Spec")
    st.dataframe(compare_df, use_container_width=True)

    st.subheader("Radar Chart — Normalised Spec Comparison")
    fig = spec_radar_chart(df, selected)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def page_depreciation(df: pd.DataFrame) -> None:
    """Page 4: Price depreciation curves by brand."""
    from analytics.price_model import fit_depreciation_curves
    st.title("📉 Price Depreciation Tracker")
    st.caption("How fast do phones lose value after launch?")

    depr = fit_depreciation_curves(df)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.dataframe(depr.set_index("brand").style.format({
            "decay_lambda": "{:.6f}",
            "price_at_6mo_pct": "{:.1f}%",
            "price_at_1yr_pct": "{:.1f}%",
            "price_at_2yr_pct": "{:.1f}%",
        }), use_container_width=True)

    with col_r:
        selected_brands = st.multiselect(
            "Brands to plot", depr["brand"].tolist(), default=depr["brand"].tolist()[:4]
        )
        if selected_brands:
            fig = depreciation_line_chart(depr[depr["brand"].isin(selected_brands)])
            st.plotly_chart(fig, use_container_width=True)

    best_retention = depr.nsmallest(1, "decay_lambda").iloc[0]
    worst_retention = depr.nlargest(1, "decay_lambda").iloc[0]
    st.markdown(f"""
    <div class="insight-box">
    🏆 <b>Best value retention:</b> {best_retention["brand"]} — retains {best_retention["price_at_1yr_pct"]}% of value after 1 year<br>
    📉 <b>Fastest depreciation:</b> {worst_retention["brand"]} — drops to {worst_retention["price_at_1yr_pct"]}% after 1 year
    </div>
    """, unsafe_allow_html=True)


# ── Main navigation ───────────────────────────────────────────────────────────
def main():
    df_full = get_data()
    filters = render_sidebar(df_full)
    df = apply_filters(df_full, filters)

    if df.empty:
        st.warning("No phones match the current filters. Try widening your selection.")
        return

    pages = {
        "📊 Market Overview": page_market_overview,
        "⚡ Spec Benchmarker": page_spec_benchmarker,
        "📉 Depreciation Tracker": page_depreciation,
    }

    page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page](df)


if __name__ == "__main__":
    main()
