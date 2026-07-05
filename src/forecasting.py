import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def find_optimal_arima(data):
    """Find optimal ARIMA parameters using auto_arima"""
    auto_model = auto_arima(
        data,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        random_state=42
    )
    return auto_model

def fit_arima(data, order):
    """Fit ARIMA model"""
    p, d, q = order
    model = ARIMA(data, order=(p, d, q))
    fit = model.fit()
    return fit

def forecast_arima(model, steps):
    """Generate ARIMA forecast"""
    forecast = model.forecast(steps=steps)
    return forecast

def calculate_metrics(actual, predicted):
    """Calculate forecast metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}