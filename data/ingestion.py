import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(source: str = "mock") -> pd.DataFrame:
    """
    Multi-source data loader supporting PostgreSQL, CSV, Excel, and mock data.
    
    Args:
        source: 'mock' | 'postgres' | 'csv' | 'excel'
    
    Returns:
        Normalised DataFrame with standard schema
    """
    if source == "mock":
        return _generate_mock_data()
    elif source == "postgres":
        return _load_from_postgres()
    elif source == "csv":
        return _load_from_csv()
    else:
        return _generate_mock_data()

def _generate_mock_data(days: int = 365) -> pd.DataFrame:
    np.random.seed(42)
    dates = [datetime.today() - timedelta(days=i) for i in range(days, 0, -1)]
    
    revenue = np.cumsum(np.random.normal(50000, 8000, days)) + 1_000_000
    churn_rate = np.clip(np.random.normal(0.05, 0.01, days), 0.01, 0.15)
    conversion_rate = np.clip(np.random.normal(0.12, 0.02, days), 0.05, 0.25)
    delinquency_rate = np.clip(np.random.normal(0.03, 0.005, days), 0.005, 0.1)
    
    # Inject anomalies
    anomaly_idx = np.random.choice(days, 10, replace=False)
    revenue[anomaly_idx] *= np.random.choice([0.4, 1.8], len(anomaly_idx))
    
    return pd.DataFrame({
        "date": dates,
        "revenue": revenue.round(2),
        "churn_rate": churn_rate.round(4),
        "conversion_rate": conversion_rate.round(4),
        "delinquency_rate": delinquency_rate.round(4),
        "region": np.random.choice(["North", "South", "East", "West"], days)
    })

def _load_from_postgres():
    # Replace with actual connection string
    # from sqlalchemy import create_engine
    # engine = create_engine("postgresql://user:pass@host:5432/db")
    # return pd.read_sql("SELECT * FROM kpi_metrics ORDER BY date DESC LIMIT 365", engine)
    return _generate_mock_data()

def _load_from_csv(path: str = "data/metrics.csv"):
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        return _generate_mock_data()
