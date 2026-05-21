import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data.ingestion import load_data
from app.anomaly import detect_anomalies
from app.forecasting import forecast_metric

st.set_page_config(page_title="Business Insights Dashboard", layout="wide", page_icon="📊")

st.title("📊 Smart Business Insights Dashboard")
st.markdown("*Real-time KPI tracking · Anomaly detection · Trend forecasting*")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    date_range = st.date_input("Date Range", [])
    region = st.multiselect("Region", ["North", "South", "East", "West", "All"], default=["All"])
    metric = st.selectbox("Primary Metric", ["Revenue", "Churn Rate", "Conversion Rate", "Delinquency Rate"])

# Load data
df = load_data()

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Revenue", f"₹{df['revenue'].sum()/1e6:.1f}M", delta=f"+{df['revenue'].pct_change().mean()*100:.1f}%")
with col2:
    st.metric("Avg Churn Rate", f"{df['churn_rate'].mean():.2%}", delta="-0.3%")
with col3:
    st.metric("Conversion Rate", f"{df['conversion_rate'].mean():.2%}", delta="+1.2%")
with col4:
    anomaly_count = len(detect_anomalies(df, metric.lower().replace(" ", "_")))
    st.metric("Anomalies Detected", anomaly_count, delta="⚠️" if anomaly_count > 0 else "✅")

st.divider()

# Trend Chart
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"{metric} Trend")
    fig = px.line(df, x="date", y=metric.lower().replace(" ", "_"),
                  title=f"{metric} over Time",
                  color_discrete_sequence=["#2563eb"])
    fig.update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("30-Day Forecast")
    forecast_df = forecast_metric(df, metric.lower().replace(" ", "_"), periods=30)
    fig2 = px.line(forecast_df, x="date", y=["actual", "forecast"],
                   color_discrete_map={"actual": "#2563eb", "forecast": "#f59e0b"})
    st.plotly_chart(fig2, use_container_width=True)

# Anomaly Table
st.subheader("🚨 Anomaly Alert Log")
anomalies = detect_anomalies(df, metric.lower().replace(" ", "_"))
if len(anomalies) > 0:
    st.dataframe(anomalies.style.highlight_max(color="lightyellow"), use_container_width=True)
else:
    st.success("No anomalies detected in the selected period.")
