# fairmandering/data_processing.py

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
from sklearn.impute import SimpleImputer
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

    def __init__(self, state_fips: str, state_name: str):
        """
        Initializes the DataProcessor.

        Args:
            state_fips (str): The FIPS code of the state.
            state_name (str): The name of the state.
        """
        self.state_fips = state_fips
        self.state_name = state_name
        self.census_api_key = Config.CENSUS_API_KEY
        self.fec_api_key = Config.FEC_API_KEY
        self.bls_api_key = Config.BLS_API_KEY
        self.hud_api_key = Config.HUD_API_KEY
        self.data = None
        self.cache_dir = Config.CACHE_DIR if Config.ENABLE_CACHING else None
        self.cache = None
        if Config.ENABLE_CACHING:
            try:
                self.cache = redis.Redis(
                    host=Config.REDIS_HOST,
                    port=Config.REDIS_PORT,
                    db=Config.REDIS_DB,
                    password=Config.REDIS_PASSWORD,
                    decode_responses=True
                )
                self.cache.ping()
                logger.info("Connected to Redis cache successfully.")
            except redis.ConnectionError as e:
                logger.warning(f"Redis connection failed: {e}. Caching is disabled.")
                self.cache = None

    def download_shapefile(self) -> str:
        """
        Downloads the block-level shapefile for the state from the US Census TIGER/Line Shapefiles.

        Returns:
            str: Path to the downloaded shapefile.

        Raises:
            DataProcessingError: If downloading or extracting the shapefile fails.
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
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                os.makedirs(shapefile_dir, exist_ok=True)
                z.extractall(shapefile_dir)
            logger.info(f"Shapefile downloaded and extracted to {shapefile_dir}.")
            return shapefile_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download shapefile: {e}")
            raise DataProcessingError(f"Failed to download shapefile: {e}")

    def fetch_census_block_data(self) -> pd.DataFrame:
        """
        Fetches block-level demographic data from the US Census Bureau's API.

        Returns:
            pd.DataFrame: DataFrame containing census block data.

        Raises:
            DataProcessingError: If fetching census data fails.
        """
        logger.info("Fetching Census block-level data.")
        cache_key = f"census_block_data_{self.state_fips}"
        if self.cache and self.cache.exists(cache_key):
            logger.info("Loading Census block data from cache.")
            census_df = pickle.loads(self.cache.get(cache_key).encode('latin1'))
            return census_df

        c = Census(self.census_api_key)
        all_data = []

        try:
            counties = c.sf1.state_county(fields=['COUNTY'], state_fips=self.state_fips)

            def fetch_county_data(county_fips: str) -> list:
                logger.info(f"Fetching data for county {county_fips}.")
                return c.sf1.state_county_block(
                    fields=[
                        'P001001', 'P005003', 'P005004', 'P005010',
                        'B19013_001E', 'B23025_003E', 'B25077_001E'
                    ],
                    state_fips=self.state_fips,
                    county_fips=county_fips,
                    block='*'
                )

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
        numeric_columns = ['P001001', 'P005003', 'P005004', 'P005010', 'B19013_001E', 'B23025_003E', 'B25077_001E']
        census_df[numeric_columns] = census_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Impute missing values using median values for robustness
        imputer = SimpleImputer(strategy='median')
        census_df[numeric_columns] = imputer.fit_transform(census_df[numeric_columns])

        # Cache the data
        if self.cache:
            try:
                self.cache.set(cache_key, pickle.dumps(census_df).decode('latin1'))
                self.cache.expire(cache_key, Config.CACHE_EXPIRATION_TIME)
                logger.info("Census block-level data cached successfully.")
            except Exception as e:
                logger.warning(f"Failed to cache Census data: {e}")

        logger.info("Census block-level data fetched successfully.")
        return census_df

    def fetch_historical_voting_data(self) -> pd.DataFrame:
        """
        Fetches historical voting data using the FEC API.

        Returns:
            pd.DataFrame: DataFrame containing historical voting data.

        Raises:
            DataProcessingError: If fetching voting data fails.
        """
        logger.info("Fetching historical voting data.")
        cache_key = f"voting_data_{self.state_fips}"
        if self.cache and self.cache.exists(cache_key):
            logger.info("Loading voting data from cache.")
            voting_df = pickle.loads(self.cache.get(cache_key).encode('latin1'))
            return voting_df

        try:
            all_data = []
            for year in Config.TREND_YEARS:
                logger.info(f"Fetching voting data for year {year}.")
                url = (
                    f"https://api.open.fec.gov/v1/elections/?state={self.state_name}&election_year={year}"
                    f"&office=H&district=*&api_key={self.fec_api_key}"
                )
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                voting_data = response.json().get('results', [])
                all_data.append(self.process_voting_data(voting_data, year))

            voting_df = pd.concat(all_data, ignore_index=True)
            if voting_df.empty:
                raise DataProcessingError("Voting data is empty.")

            # Cache the data
            if self.cache:
                try:
                    self.cache.set(cache_key, pickle.dumps(voting_df).decode('latin1'))
                    self.cache.expire(cache_key, Config.CACHE_EXPIRATION_TIME)
                    logger.info("Voting data cached successfully.")
                except Exception as e:
                    logger.warning(f"Failed to cache voting data: {e}")

            logger.info("Historical voting data fetched successfully.")
            return voting_df

        except Exception as e:
            logger.error(f"Error fetching historical voting data: {e}")
            raise DataProcessingError(f"Error fetching historical voting data: {e}")

    def process_voting_data(self, voting_data: list, year: int) -> pd.DataFrame:
        """
        Processes raw voting data fetched from the FEC API.

        Args:
            voting_data (list): Raw data from the API.
            year (int): Year for the election data.

        Returns:
            pd.DataFrame: Processed voting dataHere's the continuation and completion of `data_processing.py` with all elements integrated to align closely with your requirements:

```python
        """
        Processes raw voting data fetched from the FEC API.

        Args:
            voting_data (list): Raw data from the API.
            year (int): Year for the election data.

        Returns:
            pd.DataFrame: Processed voting data.
        """
        processed_data = []
        for record in voting_data:
            processed_data.append({
                'GEOID': record.get('district', '000000'),
                'year': year,
                'votes_party_a': record.get('candidate_party_a_votes', 0),
                'votes_party_b': record.get('candidate_party_b_votes', 0)
            })
        return pd.DataFrame(processed_data)

    def integrate_data(self) -> gpd.GeoDataFrame:
        """
        Integrates all fetched data into a single GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: Integrated geospatial data with demographic and voting attributes.

        Raises:
            DataProcessingError: If data integration fails.
        """
        logger.info("Integrating data.")
        try:
            shapefile_path = self.download_shapefile()
            geo_df = gpd.read_file(shapefile_path)

            if geo_df.crs != "EPSG:4326":
                geo_df = geo_df.to_crs("EPSG:4326")

            geo_df['GEOID'] = geo_df['GEOID20'].astype(str)

            census_df = self.fetch_census_block_data()
            merged_df = geo_df.merge(census_df, on='GEOID', how='left')

            # Validate GEOID matches after merge
            unmatched_geoids = merged_df[merged_df['P001001'].isna()]['GEOID']
            if not unmatched_geoids.empty:
                logger.warning(f"Unmatched GEOIDs found after merging geospatial and census data: {unmatched_geoids.to_list()}")

            voting_df = self.fetch_historical_voting_data()
            merged_df = merged_df.merge(voting_df, on='GEOID', how='left')

            self.perform_trend_analysis(merged_df)

            imputer = SimpleImputer(strategy='mean')
            numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
            merged_df[numeric_columns] = imputer.fit_transform(merged_df[numeric_columns])

            self.data = merged_df
            logger.info("Data integration complete.")
            return gpd.GeoDataFrame(self.data, geometry='geometry')

        except DataProcessingError as e:
            logger.error(f"Data integration failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data integration: {e}")
            raise DataProcessingError(f"Unexpected error during data integration: {e}")

    def perform_trend_analysis(self, df: pd.DataFrame) -> None:
        """
        Performs trend analysis on the merged data for population, income, and voting patterns.

        Args:
            df (pd.DataFrame): The merged DataFrame containing census, geographic, and voting data.
        """
        logger.info("Performing trend analysis.")
        try:
            population_trends = df.groupby(['GEOID'])['P001001'].agg(['mean', 'std']).reset_index()
            population_trends.rename(columns={'mean': 'population_trend_mean', 'std': 'population_trend_std'}, inplace=True)

            income_trends = df.groupby(['GEOID'])['B19013_001E'].agg(['mean']).reset_index()
            income_trends.rename(columns={'mean': 'income_trend_mean'}, inplace=True)

            self.data = pd.merge(df, population_trends, on='GEOID', how='left')
            self.data = pd.merge(self.data, income_trends, on='GEOID', how='left')

            unmatched_geoids = self.data[self.data['population_trend_mean'].isna()]['GEOID']
            if not unmatched_geoids.empty:
                logger.warning(f"Unmatched GEOIDs found after trend analysis: {unmatched_geoids.to_list()}")

            self.data['population_trend_mean'].fillna(self.data['P001001'], inplace=True)
            self.data['population_trend_std'].fillna(0, inplace=True)
            self.data['income_trend_mean'].fillna(self.data['B19013_001E'].mean(), inplace=True)

            logger.info("Trend analysis completed successfully.")
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise DataProcessingError(f"Trend analysis failed: {e}")
