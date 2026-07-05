import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

def calculate_covariance(returns_df):
    """Calculate annualized covariance matrix"""
    return returns_df.cov() * 252

def calculate_expected_returns(returns_df, method='mean'):
    """Calculate expected returns"""
    if method == 'mean':
        return returns_df.mean() * 252
    else:
        return returns_df.mean() * 252

def optimize_portfolio(mu, cov_matrix, risk_free_rate=0.02):
    """Find optimal portfolios"""
    ef_max_sharpe = EfficientFrontier(mu, cov_matrix)
    ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    max_sharpe_weights = ef_max_sharpe.clean_weights()
    ret, vol, sharpe = ef_max_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)
    
    ef_min_vol = EfficientFrontier(mu, cov_matrix)
    ef_min_vol.min_volatility()
    min_vol_weights = ef_min_vol.clean_weights()
    min_ret, min_vol, min_sharpe = ef_min_vol.portfolio_performance(risk_free_rate=risk_free_rate)
    
    return {
        'max_sharpe': {'weights': max_sharpe_weights, 'return': ret, 'volatility': vol, 'sharpe': sharpe},
        'min_volatility': {'weights': min_vol_weights, 'return': min_ret, 'volatility': min_vol, 'sharpe': min_sharpe}
    }