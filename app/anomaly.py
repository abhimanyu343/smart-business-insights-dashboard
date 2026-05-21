import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame, column: str, method: str = "zscore", threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect anomalies in time-series data using Z-score or IQR method.
    
    Args:
        df: DataFrame with date index and metric columns
        column: Column name to analyse
        method: 'zscore' or 'iqr'
        threshold: Sensitivity threshold (default 2.5 sigma)
    
    Returns:
        DataFrame of anomalous rows with deviation score
    """
    if column not in df.columns:
        return pd.DataFrame()
    
    series = df[column].dropna()
    
    if method == "zscore":
        mean, std = series.mean(), series.std()
        z_scores = np.abs((series - mean) / std)
        mask = z_scores > threshold
        df = df.loc[mask].copy()
        df["deviation_score"] = z_scores[mask].round(3)
        df["method"] = "Z-Score"
    
    elif method == "iqr":
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
        df = df.loc[mask].copy()
        df["deviation_score"] = ((series[mask] - series.median()) / IQR).abs().round(3)
        df["method"] = "IQR"
    
    return df[["date", column, "deviation_score", "method"]].sort_values("deviation_score", ascending=False)
