# tests/test_main.py

import unittest
from fairmandering.main import main
from fairmandering.config import Config

class TestMain(unittest.TestCase):
    def test_main_function(self):
        # Run the main function with test parameters
        try:
            main(state_fips=Config.STATE_FIPS, num_districts=Config.NUM_DISTRICTS)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Main function failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
