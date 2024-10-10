import os
import pandas as pd
import geopandas as gpd
import requests
from census import Census
from shapely.geometry import Point
from .config import Config
import logging
import numpy as np
from io import BytesIO
import zipfile
from us import states
from sklearn.cluster import KMeans
from datetime import datetime
from joblib import Parallel, delayed
import pickle
import hashlib
from cryptography.fernet import Fernet
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis

logger = logging.getLogger(__name__)

class DataProcessingError(Exception):
    """Custom exception class for data processing errors."""
    pass

class DataProcessor:
    """
    Responsible for fetching, processing, and integrating data from various sources.
    """

    def __init__(self, state_fips, state_name):
        self.state_fips = state_fips
        self.state_name = state_name
        self.census_api_key = Config.CENSUS_API_KEY
        self.data = None
        self.cache_dir = Config.CACHE_DIR if Config.ENABLE_CACHING else None
        self.cache = None
        if Config.ENABLE_CACHING:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=0)
                self.cache.ping()
            except redis.ConnectionError as e:
                logger.warning(f"Redis connection failed: {e}. Caching is disabled.")
                self.cache = None

    def download_shapefile(self):
        """
        Downloads the block-level shapefile for the state from the US Census TIGER/Line Shapefiles.
        """
        logger.info(f"Downloading block-level shapefile for {self.state_name}.")

        year = "2020"
        shapefile_dir = os.path.join('shapefiles', self.state_fips)
        shapefile_path = os.path.join(shapefile_dir, f"tl_{year}_{self.state_fips}_tabblock20.shp")

        if Config.ENABLE_CACHING and os.path.exists(shapefile_path):
            logger.info(f"Shapefile found in cache: {shapefile_path}")
            return shapefile_path

        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TABBLOCK20/tl_{year}_{self.state_fips}_tabblock20.zip"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            z = zipfile.ZipFile(BytesIO(response.content))
            os.makedirs(shapefile_dir, exist_ok=True)
            z.extractall(shapefile_dir)
            logger.info(f"Shapefile downloaded and extracted to {shapefile_dir}.")
            return shapefile_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download shapefile: {e}")
            raise DataProcessingError(f"Failed to download shapefile: {e}")
        except zipfile.BadZipFile as e:
            logger.error(f"Error extracting shapefile: {e}")
            raise DataProcessingError(f"Error extracting shapefile: {e}")

    def fetch_census_block_data(self):
        """
        Fetches block-level demographic data from the US Census Bureau's API.
        """
        logger.info("Fetching Census block-level data.")

        cache_key = f"census_block_data_{self.state_fips}"
        if self.cache and self.cache.exists(cache_key):
            logger.info("Loading Census block data from cache.")
            census_df = pickle.loads(self.cache.get(cache_key))
            return census_df

        c = Census(self.census_api_key)
        all_data = []

        try:
            counties = c.sf1.state_county(fields=['COUNTY'], state_fips=self.state_fips)

            def fetch_county_data(county_fips):
                logger.info(f"Fetching data for county {county_fips}.")
                data = c.sf1.state_county_block(
                    fields=[
                        'P001001',  # Total Population
                        'P005003',  # White alone
                        'P005004',  # Black or African American alone
                        'P005010',  # Hispanic or Latino
                    ],
                    state_fips=self.state_fips,
                    county_fips=county_fips,
                    block='*'
                )
                return data

            with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
                futures = [executor.submit(fetch_county_data, county['county']) for county in counties]
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        all_data.extend(data)
                    except Exception as e:
                        logger.error(f"Error fetching county data: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise DataProcessingError(f"Unexpected error: {e}")

        if not all_data:
            logger.error("No Census data retrieved.")
            raise DataProcessingError("No Census data retrieved.")

        census_df = pd.DataFrame(all_data)
        if census_df.empty:
            logger.error("Census DataFrame is empty after data retrieval.")
            raise DataProcessingError("Census DataFrame is empty after data retrieval.")

        # Convert data types and handle missing values
        census_df['GEOID'] = (
            census_df['state'] + census_df['county'] + census_df['tract'] + census_df['block']
        )
        numeric_columns = ['P001001', 'P005003', 'P005004', 'P005010']
        census_df[numeric_columns] = census_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Cache the data
        if self.cache:
            self.cache.set(cache_key, pickle.dumps(census_df))
            self.cache.expire(cache_key, Config.CACHE_EXPIRATION_TIME)

        logger.info("Census block-level data fetched successfully.")
        return census_df

    def integrate_data(self):
        """
        Integrates all fetched data into a single GeoDataFrame.
        """
        logger.info("Integrating data.")

        try:
            shapefile_path = self.download_shapefile()
            geo_df = gpd.read_file(shapefile_path)
            geo_df['GEOID'] = geo_df['GEOID20'].astype(str)

            # Fetch block-level data
            census_df = self.fetch_census_block_data()
            self.data = geo_df.merge(census_df, on='GEOID', how='left')

            # Fetch historical voting data
            voting_df = self.fetch_historical_voting_data()

            # Perform trend analysis
            self.perform_trend_analysis(census_df, voting_df)

            logger.info("Data integration complete.")
            return self.data

        except DataProcessingError as e:
            logger.error(f"Data integration failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data integration: {e}")
            raise
