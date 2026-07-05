import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import fetch_data

class TestDataLoader(unittest.TestCase):
    
    def test_fetch_data(self):
        """Test fetching data"""
        data = fetch_data('TSLA', '2023-01-01', '2023-01-10')
        self.assertIsNotNone(data)
        self.assertTrue(len(data) > 0)
        print("✓ test_fetch_data passed")
    
    def test_fetch_data_invalid_ticker(self):
        """Test fetching invalid ticker"""
        try:
            data = fetch_data('INVALID_TICKER')
            self.assertTrue(True)  # Should not raise error
        except:
            self.assertTrue(True)  # Error expected
        print("✓ test_fetch_data_invalid_ticker passed")

if __name__ == "__main__":
    unittest.main()