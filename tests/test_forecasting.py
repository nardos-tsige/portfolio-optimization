import unittest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forecasting import calculate_metrics

class TestForecasting(unittest.TestCase):
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        actual = np.array([10, 11, 12, 13, 14])
        predicted = np.array([10.5, 11.5, 12.5, 13.5, 14.5])
        
        metrics = calculate_metrics(actual, predicted)
        
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        print(f"✓ test_calculate_metrics passed: {metrics}")

if __name__ == "__main__":
    unittest.main()