"""
Data Processor Module with Validation and Error Handling
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DataProcessor:
    """Robust data processor with validation"""
    
    def __init__(self, min_rows: int = 10):
        self.min_rows = min_rows
    
    def validate_data(self, df: pd.DataFrame, required_cols: list = None) -> bool:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            required_cols: List of required columns
        
        Returns:
            True if valid, False otherwise
        """
        if df is None:
            logger.error("DataFrame is None")
            return False
        
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        if len(df) < self.min_rows:
            logger.error(f"DataFrame too small: {len(df)} rows (min: {self.min_rows})")
            return False
        
        if required_cols:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                return False
        
        return True
    
    def clean_data(self, df: pd.DataFrame, fill_method: str = 'ffill') -> Optional[pd.DataFrame]:
        """
        Clean data with error handling
        
        Args:
            df: DataFrame to clean
            fill_method: Method for filling missing values ('ffill', 'bfill', 'interpolate')
        
        Returns:
            Cleaned DataFrame or None if failed
        """
        try:
            if not self.validate_data(df):
                return None
            
            df_clean = df.copy()
            df_clean.reset_index(inplace=True)
            
            # Check for missing values
            missing = df_clean.isnull().sum()
            if missing.sum() > 0:
                logger.info(f"Found {missing.sum()} missing values")
                
                # Fill missing values based on method
                if fill_method == 'ffill':
                    df_clean = df_clean.ffill()
                elif fill_method == 'bfill':
                    df_clean = df_clean.bfill()
                elif fill_method == 'interpolate':
                    df_clean = df_clean.interpolate(method='linear')
                else:
                    logger.warning(f"Unknown fill method: {fill_method}, using ffill")
                    df_clean = df_clean.ffill()
                
                # Fill any remaining NaN with backward fill
                df_clean = df_clean.bfill()
            
            logger.info(f"Cleaned data: {len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return None
    
    def calculate_returns(self, df: pd.DataFrame, price_col: str = 'Close') -> Optional[pd.Series]:
        """
        Calculate daily returns with error handling
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
        
        Returns:
            Series of returns or None if failed
        """
        try:
            if not self.validate_data(df, [price_col]):
                return None
            
            returns = df[price_col].pct_change()
            logger.info(f"Calculated returns for {len(returns)} rows")
            return returns
            
        except KeyError:
            logger.error(f"Column '{price_col}' not found")
            return None
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return None
    
    def calculate_rolling_stats(self, df: pd.DataFrame, price_col: str = 'Close', 
                               window: int = 30) -> Optional[pd.DataFrame]:
        """
        Calculate rolling statistics with error handling
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            window: Rolling window size
        
        Returns:
            DataFrame with rolling statistics or None if failed
        """
        try:
            if not self.validate_data(df, [price_col]):
                return None
            
            stats = pd.DataFrame(index=df.index)
            stats['Rolling_Mean'] = df[price_col].rolling(window=window).mean()
            stats['Rolling_Std'] = df[price_col].rolling(window=window).std()
            
            logger.info(f"Calculated rolling stats with window={window}")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating rolling stats: {str(e)}")
            return None
    
    def detect_outliers(self, data: pd.Series, threshold: float = 3) -> Optional[pd.DataFrame]:
        """
        Detect outliers using Z-score method with error handling
        
        Args:
            data: Series to check for outliers
            threshold: Z-score threshold
        
        Returns:
            DataFrame with outliers or None if failed
        """
        try:
            if data is None or len(data) == 0:
                logger.error("Empty data for outlier detection")
                return None
            
            mean = data.mean()
            std = data.std()
            
            if std == 0:
                logger.warning("Standard deviation is zero, no outliers detected")
                return pd.DataFrame()
            
            z_scores = (data - mean) / std
            outliers = data[abs(z_scores) > threshold]
            
            logger.info(f"Found {len(outliers)} outliers")
            return outliers.to_frame('Outlier')
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return None