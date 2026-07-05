"""
Data Loader Module with Robust Error Handling
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Robust data loader with error handling and retries"""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def fetch_data(self, ticker: str, start_date: str = '2015-01-01', 
                   end_date: str = '2026-06-30') -> Optional[pd.DataFrame]:
        """
        Fetch historical data with retry logic and error handling
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with historical data or None if failed
        """
        logger.info(f"Fetching data for {ticker}...")
        
        for attempt in range(self.max_retries):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return None
                
                logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {ticker}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch {ticker} after {self.max_retries} attempts")
                    return None
        
        return None
    
    def fetch_multiple(self, tickers: list, **kwargs) -> dict:
        """
        Fetch multiple tickers with error handling
        
        Args:
            tickers: List of ticker symbols
            **kwargs: Additional arguments for fetch_data
        
        Returns:
            Dictionary of ticker -> DataFrame
        """
        result = {}
        for ticker in tickers:
            data = self.fetch_data(ticker, **kwargs)
            if data is not None:
                result[ticker] = data
            else:
                logger.warning(f"Skipping {ticker} due to fetch failure")
        
        return result

def load_processed_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load processed data with error handling
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame or None if file not found
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path, parse_dates=['Date'])
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {file_path}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error in {file_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {str(e)}")
        return None

def save_processed_data(df: pd.DataFrame, file_path: str) -> bool:
    """
    Save processed data with error handling
    
    Args:
        df: DataFrame to save
        file_path: Path to save CSV file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} rows to {file_path}")
        return True
        
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Error saving {file_path}: {str(e)}")
        return False