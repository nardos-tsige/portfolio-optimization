import os
import pandas as pd
import yfinance as yf

def fetch_data(ticker, start_date='2015-01-01', end_date='2026-06-30'):
    """Fetch historical data for a given ticker"""
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data

def load_processed_data(file_path):
    """Load processed CSV data"""
    return pd.read_csv(file_path, parse_dates=['Date'])

def save_processed_data(df, file_path):
    """Save processed data to CSV"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

if __name__ == "__main__":
    # Test the functions
    tsla = fetch_data('TSLA')
    print(f"TSLA data shape: {tsla.shape}")