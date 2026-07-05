"""
Forecasting Module with Error Handling
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ForecastingModel:
    """Robust forecasting with error handling"""
    
    def __init__(self, max_p: int = 5, max_q: int = 5, random_state: int = 42):
        self.max_p = max_p
        self.max_q = max_q
        self.random_state = random_state
        self.model = None
        self.scaler = None
    
    def find_optimal_arima(self, data: np.ndarray) -> Optional[Dict]:
        """
        Find optimal ARIMA parameters with error handling
        
        Args:
            data: Time series data
        
        Returns:
            Dictionary with optimal parameters or None if failed
        """
        try:
            if data is None or len(data) < 10:
                logger.error("Insufficient data for ARIMA optimization")
                return None
            
            # Remove NaN values
            data_clean = data[~np.isnan(data)]
            
            if len(data_clean) < 10:
                logger.error("Not enough clean data points")
                return None
            
            auto_model = auto_arima(
                data_clean,
                start_p=0, start_q=0,
                max_p=self.max_p, max_q=self.max_q,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                random_state=self.random_state
            )
            
            result = {
                'order': auto_model.order,
                'aic': auto_model.aic(),
                'model': auto_model
            }
            
            logger.info(f"Optimal ARIMA: {auto_model.order}, AIC: {auto_model.aic():.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error finding optimal ARIMA: {str(e)}")
            return None
    
    def fit_arima(self, data: np.ndarray, order: Tuple) -> Optional[Any]:
        """
        Fit ARIMA model with error handling
        
        Args:
            data: Time series data
            order: (p, d, q) tuple
        
        Returns:
            Fitted model or None if failed
        """
        try:
            if data is None or len(data) < 10:
                logger.error("Insufficient data for ARIMA fitting")
                return None
            
            data_clean = data[~np.isnan(data)]
            
            if len(data_clean) < 10:
                logger.error("Not enough clean data for fitting")
                return None
            
            model = ARIMA(data_clean, order=order)
            fit = model.fit()
            
            logger.info(f"ARIMA model fitted successfully: {order}")
            return fit
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA: {str(e)}")
            return None
    
    def forecast_arima(self, model: Any, steps: int) -> Optional[np.ndarray]:
        """
        Generate ARIMA forecast with error handling
        
        Args:
            model: Fitted ARIMA model
            steps: Number of steps to forecast
        
        Returns:
            Forecast array or None if failed
        """
        try:
            if model is None:
                logger.error("Model is None")
                return None
            
            if steps <= 0:
                logger.error(f"Invalid steps: {steps}")
                return None
            
            forecast = model.forecast(steps=steps)
            logger.info(f"Generated {len(forecast)} forecasts")
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting ARIMA: {str(e)}")
            return None
    
    def prepare_lstm_data(self, data: np.ndarray, window_size: int = 60) -> Optional[Tuple]:
        """
        Prepare data for LSTM with error handling
        
        Args:
            data: Time series data
            window_size: Lookback window size
        
        Returns:
            Tuple of (X, y, scaler) or None if failed
        """
        try:
            if data is None or len(data) < window_size:
                logger.error(f"Insufficient data for LSTM: {len(data)} < {window_size}")
                return None
            
            # Scale data
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(window_size, len(scaled_data)):
                X.append(scaled_data[i-window_size:i, 0])
                y.append(scaled_data[i, 0])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"LSTM data prepared: X={X.shape}, y={y.shape}")
            return X, y, self.scaler
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {str(e)}")
            return None
    
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Optional[Dict]:
        """
        Calculate forecast metrics with error handling
        
        Args:
            actual: Actual values
            predicted: Predicted values
        
        Returns:
            Dictionary of metrics or None if failed
        """
        try:
            if actual is None or predicted is None:
                logger.error("Actual or predicted is None")
                return None
            
            if len(actual) != len(predicted):
                logger.error(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
                return None
            
            if len(actual) == 0:
                logger.error("Empty arrays")
                return None
            
            # Remove NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) == 0:
                logger.error("No valid data points after cleaning")
                return None
            
            mae = mean_absolute_error(actual_clean, predicted_clean)
            rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
            
            # Handle zero division for MAPE
            nonzero_mask = actual_clean != 0
            if np.any(nonzero_mask):
                mape = np.mean(np.abs((actual_clean[nonzero_mask] - predicted_clean[nonzero_mask]) / actual_clean[nonzero_mask])) * 100
            else:
                mape = float('inf')
            
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'n_samples': len(actual_clean)
            }
            
            logger.info(f"Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None