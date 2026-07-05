import pandas as pd
import numpy as np

def clean_data(df):
    """Clean financial data"""
    df_clean = df.copy()
    df_clean.reset_index(inplace=True)
    df_clean = df_clean.ffill().bfill()
    return df_clean

def calculate_returns(df, price_col='Close'):
    """Calculate daily returns"""
    returns = df[price_col].pct_change()
    return returns

def calculate_rolling_stats(df, price_col='Close', window=30):
    """Calculate rolling mean and standard deviation"""
    stats = pd.DataFrame()
    stats['Rolling_Mean'] = df[price_col].rolling(window=window).mean()
    stats['Rolling_Std'] = df[price_col].rolling(window=window).std()
    return stats

def detect_outliers(data, column='Returns', threshold=3):
    """Detect outliers using Z-score method"""
    if column not in data.columns:
        return pd.DataFrame()
    mean = data[column].mean()
    std = data[column].std()
    if std == 0:
        return pd.DataFrame()
    z_scores = (data[column] - mean) / std
    outliers = data[abs(z_scores) > threshold]
    return outliers