# tests/test_visualization.py

import unittest
from fairmandering.visualization import visualize_district_map
from fairmandering.data_processing import DataProcessor
from fairmandering.optimization import optimize_districting
from fairmandering.config import Config
import os

class TestVisualization(unittest.TestCase):
    def setUp(self):
        processor = DataProcessor(Config.STATE_FIPS, Config.STATE_NAME)
        self.data = processor.integrate_data()
        assignments, _ = optimize_districting(self.data, seeds=[1])
        self.assignment = assignments[0]

    def test_visualize_district_map(self):
        visualize_district_map(self.data, self.assignment)
        self.assertTrue(os.path.exists('district_map.html'))

if __name__ == '__main__':
    unittest.main()
