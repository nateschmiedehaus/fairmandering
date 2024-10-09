# tests/test_data_processing.py

import unittest
from fairmandering.data_processing import DataProcessor, DataProcessingError
from fairmandering.config import Config

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor(Config.STATE_FIPS, Config.STATE_NAME)

    def test_download_shapefile(self):
        shapefile_path = self.processor.download_shapefile()
        self.assertTrue(os.path.exists(shapefile_path))

    def test_fetch_census_block_data(self):
        census_df = self.processor.fetch_census_block_data()
        self.assertFalse(census_df.empty)

    def test_integrate_data(self):
        data = self.processor.integrate_data()
        self.assertFalse(data.empty)
        self.assertIn('GEOID', data.columns)

if __name__ == '__main__':
    unittest.main()
