"""
Backtesting script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.portfolio import optimize_portfolio

def calculate_portfolio_returns(weights, returns_df):
    """Calculate portfolio returns"""
    portfolio_returns = pd.Series(0, index=returns_df.index)
    for asset, weight in weights.items():
        if weight > 0:
            portfolio_returns += weight * returns_df[asset]
    return portfolio_returns

def calculate_metrics(returns):
    """Calculate performance metrics"""
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252/len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Max drawdown
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

def run_backtest(weights, returns_df, benchmark_weights=None):
    """Run backtest"""
    print("Running backtest...")
    
    # Calculate strategy returns
    strategy_returns = calculate_portfolio_returns(weights, returns_df)
    strategy_metrics = calculate_metrics(strategy_returns)
    
    if benchmark_weights:
        benchmark_returns = calculate_portfolio_returns(benchmark_weights, returns_df)
        benchmark_metrics = calculate_metrics(benchmark_returns)
        
        print("\nStrategy vs Benchmark:")
        print(f"Strategy Return: {strategy_metrics['total_return']:.2%}")
        print(f"Benchmark Return: {benchmark_metrics['total_return']:.2%}")
        print(f"Outperformance: {strategy_metrics['total_return'] - benchmark_metrics['total_return']:.2%}")
    
    return strategy_metrics

if __name__ == "__main__":
    # Test backtest
    print("Testing backtest...")