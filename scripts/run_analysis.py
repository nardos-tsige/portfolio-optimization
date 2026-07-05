"""
Main script to run all analyses
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from src.data_loader import fetch_data
from src.data_processor import clean_data, calculate_returns
from src.forecasting import find_optimal_arima, fit_arima, forecast_arima

def main():
    print("=" * 60)
    print("RUNNING COMPLETE ANALYSIS")
    print("=" * 60)
    
    # 1. Fetch data
    print("\n1. Fetching data...")
    tsla = fetch_data('TSLA')
    bnd = fetch_data('BND')
    spy = fetch_data('SPY')
    
    # 2. Clean data
    print("\n2. Cleaning data...")
    tsla_clean = clean_data(tsla)
    bnd_clean = clean_data(bnd)
    spy_clean = clean_data(spy)
    
    # 3. Calculate returns
    print("\n3. Calculating returns...")
    tsla_returns = calculate_returns(tsla_clean)
    
    print("\nAnalysis complete!")
    print(f"TSLA data shape: {tsla_clean.shape}")
    print(f"TSLA returns shape: {tsla_returns.shape}")

if __name__ == "__main__":
    main()