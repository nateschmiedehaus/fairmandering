# tests/test_optimization.py

import unittest
from fairmandering.optimization import optimize_districting
from fairmandering.data_processing import DataProcessor
from fairmandering.config import Config

class TestOptimization(unittest.TestCase):
    def setUp(self):
        processor = DataProcessor(Config.STATE_FIPS, Config.STATE_NAME)
        self.data = processor.integrate_data()

    def test_optimize_districting(self):
        assignments, objectives = optimize_districting(self.data, seeds=[1])
        self.assertTrue(len(assignments) > 0)
        self.assertTrue(len(objectives) > 0)

if __name__ == '__main__':
    unittest.main()
