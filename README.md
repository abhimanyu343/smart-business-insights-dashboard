# Smart Business Insights Dashboard

> Automated KPI tracking, anomaly detection, and revenue trend analysis — built with Python, Streamlit, SQL, and Power BI-style visuals.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![SQL](https://img.shields.io/badge/SQL-PostgreSQL-blue) ![Pandas](https://img.shields.io/badge/Pandas-2.0-green)

## Overview

Built from real-world experience managing CCAR credit analytics and manufacturing KPI dashboards. This project automates the full analytics pipeline — from raw data ingestion to executive-ready dashboards — reducing manual reporting time by ~60%.

## Features

- **KPI Dashboard** — Revenue, churn, delinquency, and conversion metrics in real time
- **Anomaly Detection** — Z-score + IQR based alerts for metric deviations
- **Trend Forecasting** — ARIMA-based 30/60/90-day projections
- **Automated Reports** — Scheduled PDF/Excel exports via APScheduler
- **Multi-source ingestion** — PostgreSQL, CSV, Excel, REST APIs

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | PostgreSQL, Pandas, SQLAlchemy |
| ML | Scikit-learn, Statsmodels (ARIMA) |
| Viz | Plotly, Matplotlib, Seaborn |
| App | Streamlit, FastAPI |
| Automation | APScheduler, Python-pptx |

## Project Structure

```
smart-business-insights-dashboard/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── dashboard.py         # KPI dashboard components
│   ├── anomaly.py           # Anomaly detection logic
│   └── forecasting.py       # ARIMA forecasting module
├── data/
│   ├── ingestion.py         # Multi-source data loader
│   └── transforms.py        # ETL transformations
├── reports/
│   └── report_generator.py  # Automated PDF/Excel export
├── config.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/abhimanyu343/smart-business-insights-dashboard
cd smart-business-insights-dashboard
pip install -r requirements.txt

# Set your DB credentials in config.py
python -m streamlit run app/main.py
```

## Screenshots

Dashboard renders live KPIs with drill-down capability by region, product, and time period.

## Background

Inspired by real analytics work at EXL Service (credit CCAR reporting) and GPIL (manufacturing KPI automation). The ETL pipeline design mirrors production-grade patterns used in enterprise environments.

---
*Part of my AI & Data Engineering portfolio — [LinkedIn](https://linkedin.com/in/abhimanyusarda343)*
