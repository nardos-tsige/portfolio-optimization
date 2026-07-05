"""
Portfolio Optimization Module with Error Handling
"""
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions
import logging
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Robust portfolio optimizer with error handling"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.cov_matrix = None
        self.mu = None
    
    def calculate_covariance(self, returns_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate covariance matrix with error handling
        
        Args:
            returns_df: DataFrame of asset returns
        
        Returns:
            Annualized covariance matrix or None if failed
        """
        try:
            if returns_df is None or returns_df.empty:
                logger.error("Returns DataFrame is empty or None")
                return None
            
            if len(returns_df.columns) < 2:
                logger.error("Need at least 2 assets for covariance")
                return None
            
            cov_matrix = returns_df.cov() * 252
            logger.info(f"Covariance matrix calculated for {len(returns_df.columns)} assets")
            return cov_matrix
            
        except Exception as e:
            logger.error(f"Error calculating covariance: {str(e)}")
            return None
    
    def calculate_expected_returns(self, returns_df: pd.DataFrame, 
                                  method: str = 'mean') -> Optional[pd.Series]:
        """
        Calculate expected returns with error handling
        
        Args:
            returns_df: DataFrame of asset returns
            method: 'mean' or 'forecast'
        
        Returns:
            Expected returns series or None if failed
        """
        try:
            if returns_df is None or returns_df.empty:
                logger.error("Returns DataFrame is empty or None")
                return None
            
            if method == 'mean':
                expected = returns_df.mean() * 252
            else:
                expected = returns_df.mean() * 252
            
            logger.info(f"Expected returns calculated: {expected.to_dict()}")
            return expected
            
        except Exception as e:
            logger.error(f"Error calculating expected returns: {str(e)}")
            return None
    
    def optimize_portfolio(self, mu: np.ndarray, cov_matrix: pd.DataFrame,
                          target_return: Optional[float] = None) -> Optional[Dict]:
        """
        Optimize portfolio with error handling
        
        Args:
            mu: Expected returns array
            cov_matrix: Covariance matrix
            target_return: Optional target return for optimization
        
        Returns:
            Dictionary with optimization results or None if failed
        """
        try:
            if mu is None or cov_matrix is None:
                logger.error("Mu or covariance matrix is None")
                return None
            
            if len(mu) != len(cov_matrix):
                logger.error(f"Dimension mismatch: mu={len(mu)}, cov={len(cov_matrix)}")
                return None
            
            results = {}
            
            # Maximum Sharpe Ratio
            try:
                ef_sharpe = EfficientFrontier(mu, cov_matrix)
                ef_sharpe.max_sharpe(risk_free_rate=self.risk_free_rate)
                max_sharpe_weights = ef_sharpe.clean_weights()
                max_ret, max_vol, max_sharpe = ef_sharpe.portfolio_performance(
                    risk_free_rate=self.risk_free_rate
                )
                
                results['max_sharpe'] = {
                    'weights': max_sharpe_weights,
                    'return': max_ret,
                    'volatility': max_vol,
                    'sharpe': max_sharpe
                }
                logger.info(f"Max Sharpe portfolio: {max_sharpe:.4f}")
                
            except Exception as e:
                logger.warning(f"Error calculating max Sharpe: {str(e)}")
                results['max_sharpe'] = {'error': str(e)}
            
            # Minimum Volatility
            try:
                ef_min_vol = EfficientFrontier(mu, cov_matrix)
                ef_min_vol.min_volatility()
                min_vol_weights = ef_min_vol.clean_weights()
                min_ret, min_vol, min_sharpe = ef_min_vol.portfolio_performance(
                    risk_free_rate=self.risk_free_rate
                )
                
                results['min_volatility'] = {
                    'weights': min_vol_weights,
                    'return': min_ret,
                    'volatility': min_vol,
                    'sharpe': min_sharpe
                }
                logger.info(f"Min Volatility portfolio: {min_vol:.4f}")
                
            except Exception as e:
                logger.warning(f"Error calculating min volatility: {str(e)}")
                results['min_volatility'] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return None
    
    def generate_efficient_frontier(self, mu: np.ndarray, cov_matrix: pd.DataFrame,
                                   n_points: int = 50) -> Optional[Dict]:
        """
        Generate efficient frontier points with error handling
        
        Args:
            mu: Expected returns array
            cov_matrix: Covariance matrix
            n_points: Number of points to generate
        
        Returns:
            Dictionary with frontier points or None if failed
        """
        try:
            if mu is None or cov_matrix is None:
                logger.error("Mu or covariance matrix is None")
                return None
            
            returns = []
            volatilities = []
            weights_list = []
            
            min_ret = mu.min()
            max_ret = mu.max()
            
            for target_return in np.linspace(min_ret, max_ret, n_points):
                try:
                    ef = EfficientFrontier(mu, cov_matrix)
                    ef.efficient_return(target_return)
                    ret, vol, _ = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
                    
                    returns.append(ret)
                    volatilities.append(vol)
                    weights_list.append(ef.weights)
                    
                except Exception:
                    continue
            
            if not returns:
                logger.warning("No efficient frontier points generated")
                return None
            
            logger.info(f"Generated {len(returns)} efficient frontier points")
            return {
                'returns': returns,
                'volatilities': volatilities,
                'weights': weights_list
            }
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
            return None