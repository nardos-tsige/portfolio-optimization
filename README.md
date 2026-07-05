# Portfolio Optimization with Time Series Forecasting

## Project Overview
This project implements a comprehensive portfolio management system using time series forecasting and Modern Portfolio Theory (MPT). It fetches historical financial data for three key assets (TSLA, BND, SPY), builds forecasting models, and creates optimized portfolios based on predicted returns.

**For:** GMF Investments - 10 Academy Artificial Intelligence Mastery (Week 9 Challenge)
**Date:** July 1 - July 7, 2026

---

## Objectives
- Extract and preprocess historical financial data from YFinance
- Build and compare ARIMA and LSTM forecasting models
- Generate 6-month future price forecasts with confidence intervals
- Optimize portfolio allocation using Modern Portfolio Theory
- Backtest strategy performance against a benchmark

---

## Project Structure
portfolio-optimization/
├── data/
│   ├── processed/               # Cleaned data and visualizations
│   │   ├── *.csv                # 11 processed data files
│   │   └── *.png                # 14 visualization outputs
│   └── README.md                # Data folder documentation
│
├── notebooks/                    # Jupyter notebooks
│   ├── task1_data_exploration.ipynb
│   ├── task2_forecasting_models.ipynb
│   ├── task3_forecast_future_trends.ipynb
│   ├── task4_portfolio_optimization.ipynb
│   └── task5_backtesting.ipynb
│
├── src/                          # Source code modules
│   ├── data_loader.py            # Data fetching functions
│   ├── data_processor.py         # Data cleaning and processing
│   ├── forecasting.py            # ARIMA/LSTM models
│   └── portfolio.py              # Portfolio optimization
│
├── scripts/                      # Executable scripts
│   ├── run_analysis.py           # Main analysis runner
│   └── backtest.py               # Backtesting engine
│
├── tests/                        # Unit tests
│   ├── test_data_loader.py
│   └── test_forecasting.py
│
├── reports/                       # Documentation
│   ├── interim_reports/
│   └── final_report/
│
├── .github/workflows/             # CI/CD
│   └── unittests.yml              # GitHub Actions workflow
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file

---

## Technologies Used

| Category | Tools/Libraries |
|----------|-----------------|
| **Data Fetching** | yfinance |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Time Series** | statsmodels, pmdarima |
| **Deep Learning** | tensorflow, keras |
| **Portfolio Optimization** | PyPortfolioOpt, scipy |
| **Testing** | pytest |
| **Notebooks** | Jupyter |

---

## Assets Analyzed

| Asset | Ticker | Description | Risk Profile |
|-------|--------|-------------|--------------|
| Tesla | TSLA | High-growth stock | High risk, high return |
| Vanguard Total Bond Market ETF | BND | U.S. investment-grade bonds | Low risk, stability |
| S&P 500 ETF | SPY | Broad market exposure | Moderate risk, diversification |

**Period:** January 1, 2015 - June 30, 2026

---

## Tasks Completed

### Task 1: Data Preprocessing and Exploration
- Downloaded data from YFinance
- Cleaned and handled missing values
- Generated visualizations:
  - Closing prices over time
  - Daily returns distribution
  - Rolling statistics (mean & volatility)
- Performed stationarity testing (ADF test)
- Calculated risk metrics: VaR, Sharpe Ratio

### Task 2: Time Series Forecasting Models
- **ARIMA Model:**
  - Used auto_arima for optimal parameters
  - Generated test period forecasts
  - Performance: MAE, RMSE, MAPE
- **LSTM Model:**
  - Built with 2 LSTM layers + Dropout
  - Trained on 60-day windows
  - Generated predictions and compared with ARIMA

### Task 3: Future Market Trends Forecast
- Generated 6-month (180-day) forecast
- Included 95% confidence intervals
- Performed trend analysis (upward/downward)
- Assessed market opportunities and risks
- Critical assessment of forecast reliability

### Task 4: Portfolio Optimization
- Used forecasted returns for TSLA
- Used historical returns for BND and SPY
- Calculated covariance matrix
- Generated Efficient Frontier
- Identified key portfolios:
  - **Maximum Sharpe Ratio** (Tangency Portfolio)
  - **Minimum Volatility** portfolio
- Provided final portfolio recommendation

### Task 5: Strategy Backtesting
- Backtested strategy against 60/40 benchmark
- Simulated with and without monthly rebalancing
- Calculated performance metrics:
  - Total Return
  - Annualized Return
  - Volatility
  - Sharpe Ratio
  - Maximum Drawdown
- Generated cumulative returns comparison plots

---

## Model Performance Comparison

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | $X.XX | $X.XX | X.X% |
| LSTM | $X.XX | $X.XX | X.X% |

*[Replace with actual values from your model_comparison.csv]*

---

## Portfolio Recommendation

| Portfolio | TSLA | BND | SPY | Expected Return | Volatility | Sharpe Ratio |
|-----------|------|-----|-----|-----------------|------------|--------------|
| Max Sharpe | XX% | XX% | XX% | XX.X% | XX.X% | X.XXX |
| Min Volatility | XX% | XX% | XX% | XX.X% | XX.X% | X.XXX |

**Recommended:** [Max Sharpe Ratio / Min Volatility] Portfolio

---

## Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/portfolio-optimization.git
cd portfolio-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Notebooks
```bash
jupyter notebook notebooks/
```

### Run Scripts
```bash
# Run complete analysis
python scripts/run_analysis.py

# Run backtest
python scripts/backtest.py
```

### Run Tests
```bash
pytest tests/
```

---

## Key Visualizations

| Visualization | Description |
|----------------|--------------|
| closing_prices.png | Asset price trends over time |
| daily_returns.png | Returns distribution and volatility |
| arima_forecast.png | ARIMA model predictions |
| lstm_prediction.png | LSTM model predictions |
| future_forecast_with_ci.png | 6-month forecast with confidence intervals |
| efficient_frontier.png | Optimal portfolios visualization |
| backtest_comparison.png | Strategy vs benchmark performance |

---

## Data Files

| File | Description |
|------|--------------|
| TSLA_processed.csv | Tesla stock data |
| BND_processed.csv | Bond ETF data |
| SPY_processed.csv | S&P 500 ETF data |
| all_assets_combined.csv | Combined data |
| risk_metrics.csv | VaR, Sharpe Ratio metrics |
| model_comparison.csv | ARIMA vs LSTM performance |
| future_forecast.csv | 6-month forecast with confidence intervals |
| portfolio_recommendations.csv | Optimal portfolio weights |
| backtest_metrics.csv | Strategy vs benchmark comparison |

---

## Team

| Name | Role |
|------|------|
| Kerod | Lead |
| Mahbubah | Team Member |
| Feven | Team Member |

---

## Timeline

| Date | Event |
|------|-------|
| July 1, 2026 | Challenge Introduction |
| July 5, 2026 | Interim Submission |
| July 7, 2026 | Final Submission |

---

## References

**Time Series**
- ARIMA Tutorial - DataCamp
- ARIMA for Time Series - Machine Learning Mastery

**Portfolio Optimization**
- PyPortfolioOpt Documentation
- Modern Portfolio Theory

**LSTM**
- LSTM for Time Series - TensorFlow
- LSTM Stock Market - DataCamp

---

## License
MIT License

---

## Contact
For questions or feedback, please reach out via the course Slack channel: #all-week9

---

## Acknowledgments
- GMF Investments for the business context
- 10 Academy for the challenge framework
- Open-source libraries used in this project

---

Built for GMF Investments | Week 9 Challenge
