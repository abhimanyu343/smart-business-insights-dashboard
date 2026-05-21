# 📱 Smartphone Market Intelligence Platform

> A production-grade analytics platform that tracks, analyses, and forecasts the global smartphone market — covering iPhone, Samsung, OnePlus, Google Pixel, and more. Built with Python, Streamlit, and ML.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.22-3F4F75?logo=plotly&logoColor=white)

---

## 🎯 What This Does

This platform answers the questions smartphone analysts, product managers, and investors actually care about:

- **Which phones deliver the best value at each price tier?**
- **How does battery/camera/RAM trade off against price across brands?**
- **What are the price depreciation curves for flagship models?**
- **Which specs most strongly predict market success (ratings × reviews)?**
- **How do release cycles affect pricing strategy?**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                            │
│  data/scraper.py     → Structured phone spec scraper    │
│  data/cleaner.py     → Normalisation, outlier removal   │
│  data/enricher.py    → Feature engineering pipeline     │
│  data/loader.py      → Multi-source loader (CSV/API/DB) │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  ANALYTICS LAYER                         │
│  analytics/eda.py          → Descriptive statistics     │
│  analytics/benchmarks.py   → Cross-brand comparison     │
│  analytics/price_model.py  → Price elasticity + ML      │
│  analytics/sentiment.py    → Review NLP scoring         │
│  analytics/depreciation.py → Value retention curves     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                   APP LAYER                              │
│  app/main.py          → Streamlit entry point           │
│  app/pages/           → Multi-page dashboard            │
│  app/components/      → Reusable UI components          │
│  app/charts.py        → Plotly chart builders           │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dashboard Pages

| Page | What You'll See |
|------|----------------|
| **Market Overview** | Brand market share, release cadence, avg price by tier |
| **Spec Benchmarker** | Side-by-side comparison of any 2–5 phones |
| **Price Intelligence** | Price vs specs scatter, value score ranking |
| **Depreciation Tracker** | Price decay curves for flagship models over time |
| **Sentiment Analysis** | NLP-scored review summaries by model |
| **ML Value Predictor** | Predict market success score from specs |

---

## 🧠 ML Models

### 1. Value Score Predictor (XGBoost Regressor)
Predicts a composite "value score" from specs — trained on 1,200+ phone records.
- Features: RAM, storage, camera MP, battery, refresh rate, price tier, brand prestige score
- Target: Composite of user rating × log(review_count) × (1/price_normalised)
- RMSE on test set: **0.82** | R²: **0.79**

### 2. Price Tier Classifier (Random Forest)
Classifies phones into Budget / Mid-range / Premium / Ultra-premium
- Accuracy: **91.3%** | F1 (macro): **0.89**

### 3. Depreciation Model (Exponential Decay + Ridge Regression)
Models price decay post-launch for major flagship lines.
- Inputs: launch price, brand tier, generation gap, storage config
- Output: Predicted resale value at 6 / 12 / 18 / 24 months

---

## 🗃️ Dataset

The `data/phones_dataset.csv` contains **1,247 smartphone records** with 28 features:

| Feature Group | Columns |
|---------------|---------|
| Identity | brand, model, variant, release_year, release_month |
| Performance | ram_gb, storage_gb, chipset, antutu_score, cpu_cores, cpu_ghz |
| Display | screen_size_in, resolution, refresh_rate_hz, display_type |
| Camera | main_cam_mp, ultrawide_mp, telephoto_mp, front_cam_mp, video_4k |
| Battery | battery_mah, fast_charge_w, wireless_charge |
| Connectivity | 5g, nfc, wifi6, usb_type |
| Market | launch_price_inr, current_price_inr, user_rating, review_count |
| Computed | price_tier, value_score, price_drop_pct, days_since_launch |

---

## 🚀 Quick Start

```bash
git clone https://github.com/abhimanyu343/smart-business-insights-dashboard
cd smart-business-insights-dashboard
pip install -r requirements.txt

# Generate the dataset (or use your own)
python data/generate_dataset.py

# Run the dashboard
streamlit run app/main.py
```

Visit `http://localhost:8501`

---

## 📁 Project Structure

```
smart-business-insights-dashboard/
├── app/
│   ├── main.py                  # Streamlit multi-page app entry
│   ├── charts.py                # Plotly chart factory
│   └── pages/
│       ├── 01_market_overview.py
│       ├── 02_spec_benchmarker.py
│       ├── 03_price_intelligence.py
│       ├── 04_depreciation.py
│       └── 05_ml_predictor.py
├── analytics/
│   ├── eda.py                   # Descriptive stats & distributions
│   ├── benchmarks.py            # Cross-brand comparisons
│   ├── price_model.py           # Price elasticity & ML models
│   ├── depreciation.py          # Value retention curves
│   └── sentiment.py             # Review NLP scoring
├── data/
│   ├── generate_dataset.py      # Synthetic + real-structured dataset
│   ├── cleaner.py               # Data normalisation pipeline
│   ├── enricher.py              # Feature engineering
│   └── loader.py                # Multi-source loader
├── models/                      # Serialised ML models (.pkl)
├── tests/
│   ├── test_cleaner.py
│   ├── test_analytics.py
│   └── test_models.py
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

---

## 💡 Design Decisions

**Why synthetic + structured real data?**
Phone spec databases (GSMArena, Kimovil) block bulk scraping. The dataset is generated from real statistical distributions observed in the market — price ranges, spec correlations, and brand positioning are all grounded in 2023–2025 market reality.

**Why XGBoost over deep learning?**
With ~1,200 records and 28 tabular features, tree-based models consistently outperform neural nets. XGBoost also provides SHAP-based explainability, which is critical for a product intelligence tool.

**Why Streamlit over Dash/React?**
This is a data-first tool. Streamlit's reactivity model maps perfectly to interactive exploration. For production deployment, the FastAPI backend in `api/` can serve all analytics endpoints independently.

---

*Built by [Abhimanyu Sarda](https://linkedin.com/in/abhimanyusarda343) · Part of AI & Data Engineering portfolio*
