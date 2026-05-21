import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def forecast_metric(df: pd.DataFrame, column: str, periods: int = 30) -> pd.DataFrame:
    """
    ARIMA-based time-series forecasting for business metrics.
    
    Args:
        df: DataFrame with 'date' column and metric column
        column: Metric to forecast
        periods: Number of future periods to forecast
    
    Returns:
        DataFrame with actual + forecast values
    """
    series = df.set_index("date")[column].dropna()
    
    try:
        model = ARIMA(series, order=(2, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=periods)
        
        future_dates = pd.date_range(start=series.index[-1], periods=periods + 1, freq="D")[1:]
        
        actual_df = pd.DataFrame({"date": series.index, "actual": series.values, "forecast": np.nan})
        forecast_df = pd.DataFrame({"date": future_dates, "actual": np.nan, "forecast": forecast.values})
        
        return pd.concat([actual_df, forecast_df], ignore_index=True)
    
    except Exception:
        # Fallback: simple moving average projection
        ma = series.rolling(7).mean().iloc[-1]
        future_dates = pd.date_range(start=series.index[-1], periods=periods + 1, freq="D")[1:]
        actual_df = pd.DataFrame({"date": series.index, "actual": series.values, "forecast": np.nan})
        forecast_df = pd.DataFrame({"date": future_dates, "actual": np.nan, "forecast": [ma] * periods})
        return pd.concat([actual_df, forecast_df], ignore_index=True)
