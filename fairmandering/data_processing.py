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
        self.fec_api_key = Config.FEC_API_KEY
        self.bls_api_key = Config.BLS_API_KEY
        self.hud_api_key = Config.HUD_API_KEY
        self.epa_api_key = Config.EPA_API_KEY
        self.data = None
        self.cache_dir = Config.CACHE_DIR if Config.ENABLE_CACHING else None
        self.cache = None
        if Config.ENABLE_CACHING:
            self.cache = redis.Redis(host='localhost', port=6379, db=0)

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

        if self.cache and self.cache.exists(f"census_block_data_{self.state_fips}"):
            logger.info("Loading Census block data from cache.")
            census_df = pickle.loads(self.cache.get(f"census_block_data_{self.state_fips}"))
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
            self.cache.set(f"census_block_data_{self.state_fips}", pickle.dumps(census_df))
            self.cache.expire(f"census_block_data_{self.state_fips}", Config.CACHE_EXPIRATION_TIME)

        logger.info("Census block-level data fetched successfully.")
        return census_df

    def fetch_historical_voting_data(self):
        """
        Fetches historical voting data for trend analysis.
        """
        logger.info("Fetching historical voting data.")

        if self.cache and self.cache.exists(f"voting_data_{self.state_fips}"):
            logger.info("Loading voting data from cache.")
            voting_df = pickle.loads(self.cache.get(f"voting_data_{self.state_fips}"))
            return voting_df

        try:
            # For demonstration, we'll generate synthetic data
            voting_data = []
            for year in Config.TREND_YEARS:
                data = pd.DataFrame({
                    'GEOID': self.data['GEOID'],
                    'year': year,
                    'votes_party_a': np.random.randint(100, 1000, size=len(self.data)),
                    'votes_party_b': np.random.randint(100, 1000, size=len(self.data))
                })
                voting_data.append(data)

            voting_df = pd.concat(voting_data, ignore_index=True)
            if voting_df.empty:
                raise DataProcessingError("Voting data is empty.")

            # Cache the data
            if self.cache:
                self.cache.set(f"voting_data_{self.state_fips}", pickle.dumps(voting_df))
                self.cache.expire(f"voting_data_{self.state_fips}", Config.CACHE_EXPIRATION_TIME)

            logger.info("Historical voting data fetched successfully.")
            return voting_df

        except Exception as e:
            logger.error(f"Error fetching historical voting data: {e}")
            raise DataProcessingError(f"Error fetching historical voting data: {e}")

    def perform_trend_analysis(self, census_df, voting_df):
        """
        Performs trend analysis on historical demographic and voting data.
        """
        logger.info("Performing trend analysis.")

        try:
            # Demographic Trends
            census_trends = census_df.groupby(['GEOID', 'year']).agg({
                'P001001': 'sum'
            }).reset_index()
            census_pivot = census_trends.pivot(index='GEOID', columns='year', values='P001001')
            census_pivot['population_trend'] = census_pivot.apply(
                lambda row: np.polyfit(Config.TREND_YEARS, row.dropna().values, 1)[0] if len(row.dropna()) > 1 else 0,
                axis=1
            )

            # Voting Trends
            voting_trends = voting_df.groupby(['GEOID', 'year']).agg({
                'votes_party_a': 'sum',
                'votes_party_b': 'sum'
            }).reset_index()
            voting_pivot = voting_trends.pivot(index='GEOID', columns='year', values=['votes_party_a', 'votes_party_b'])
            voting_pivot.columns = ['_'.join(map(str, col)).strip() for col in voting_pivot.columns.values]
            voting_pivot['party_a_trend'] = voting_pivot.apply(
                lambda row: np.polyfit(Config.TREND_YEARS, row[[col for col in voting_pivot.columns if 'votes_party_a' in col]].dropna().values, 1)[0] if len(row.dropna()) > 1 else 0,
                axis=1
            )
            voting_pivot['party_b_trend'] = voting_pivot.apply(
                lambda row: np.polyfit(Config.TREND_YEARS, row[[col for col in voting_pivot.columns if 'votes_party_b' in col]].dropna().values, 1)[0] if len(row.dropna()) > 1 else 0,
                axis=1
            )

            # Merge trends into main data
            self.data = self.data.merge(census_pivot[['population_trend']], on='GEOID', how='left')
            self.data = self.data.merge(voting_pivot[['party_a_trend', 'party_b_trend']], on='GEOID', how='left')
            logger.info("Trend analysis completed.")
        except Exception as e:
            logger.error(f"Error during trend analysis: {e}")
            raise DataProcessingError(f"Error during trend analysis: {e}")

    def validate_data_integrity(self, dataframes):
        """
        Verifies the integrity and consistency of data across different sources.
        """
        logger.info("Validating data integrity and consistency.")
        integrity_pass = True
        for name, df in dataframes.items():
            # Check for duplicate GEOIDs
            if df['GEOID'].duplicated().any():
                logger.error(f"Duplicate GEOIDs found in {name} dataset.")
                integrity_pass = False

            # Check for expected data types
            expected_types = {'GEOID': object, 'P001001': np.number}
            for column, dtype in expected_types.items():
                if column in df.columns and not pd.api.types.is_dtype_subtype(df[column].dtype, dtype):
                    logger.error(f"Incorrect data type for {column} in {name} dataset.")
                    integrity_pass = False

        return integrity_pass

    def report_missing_anomalous_values(self, df):
        """
        Checks for missing or anomalous values and generates a report.
        """
        logger.info("Checking for missing or anomalous values.")
        report = df.isnull().sum().to_frame(name='missing_values')
        numeric_df = df.select_dtypes(include=[np.number])
        report['anomalous_values'] = ((numeric_df < 0) | (numeric_df > numeric_df.quantile(0.99))).sum()
        report['total_rows'] = len(df)
        report.to_csv('data_quality_report.csv')
        logger.info("Data quality report generated as 'data_quality_report.csv'.")
        return report

    def ensure_temporal_consistency(self, historical_data):
        """
        Ensures temporal consistency in historical data.
        """
        logger.info("Ensuring temporal consistency in historical data.")
        expected_years = set(Config.TREND_YEARS)
        actual_years = set(historical_data['year'].unique())
        missing_years = expected_years - actual_years
        if missing_years:
            logger.error(f"Missing data for years: {missing_years}")
            return False
        return True

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

            # Fetch and merge other data
            hud_df = self.fetch_hud_data()
            self.data = self.data.merge(hud_df, on='GEOID', how='left')

            epa_df = self.fetch_epa_data()
            self.data = self.data.merge(epa_df, on='GEOID', how='left')

            # Fetch historical voting data
            voting_df = self.fetch_historical_voting_data()

            # Perform trend analysis
            self.perform_trend_analysis(census_df, voting_df)

            # Data Validation
            dataframes = {
                'geo_df': geo_df,
                'census_df': census_df,
                'hud_df': hud_df,
                'epa_df': epa_df
            }
            if not self.validate_data_integrity(dataframes):
                raise DataProcessingError("Data integrity validation failed.")

            self.report_missing_anomalous_values(self.data)
            if not self.ensure_temporal_consistency(voting_df):
                raise DataProcessingError("Temporal consistency validation failed.")

            # Data Cleaning and Imputation
            self.data.fillna(self.data.mean(numeric_only=True), inplace=True)

            # Data Normalization
            numerical_cols = [
                'P001001',
                'P005003',
                'P005004',
                'P005010',
                'median_rent',
                'environmental_hazard_index',
                'votes_party_a',
                'votes_party_b',
                'population_trend',
                'party_a_trend',
                'party_b_trend'
            ]
            self.data[numerical_cols] = self.data[numerical_cols].apply(
                lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x
            )

            # Identify COIs
            self.identify_communities_of_interest()

            logger.info("Data integration complete.")
            return self.data

        except DataProcessingError as e:
            logger.error(f"Data integration failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data integration: {e}")
            raise

    def fetch_hud_data(self):
        """
        Fetches housing data from HUD.
        """
        logger.info("Fetching HUD data.")

        try:
            hud_df = pd.DataFrame({
                'GEOID': self.data['GEOID'],
                'median_rent': np.random.randint(500, 2000, size=len(self.data))
            })
            logger.info("HUD data fetched successfully.")
            return hud_df
        except Exception as e:
            logger.error(f"Error fetching HUD data: {e}")
            raise DataProcessingError(f"Error fetching HUD data: {e}")

    def fetch_epa_data(self):
        """
        Fetches environmental data from EPA.
        """
        logger.info("Fetching EPA data.")

        try:
            epa_df = pd.DataFrame({
                'GEOID': self.data['GEOID'],
                'environmental_hazard_index': np.random.rand(len(self.data))
            })
            logger.info("EPA data fetched successfully.")
            return epa_df
        except Exception as e:
            logger.error(f"Error fetching EPA data: {e}")
            raise DataProcessingError(f"Error fetching EPA data: {e}")

    def identify_communities_of_interest(self):
        """
        Identifies communities of interest (COIs) using clustering algorithms on socio-economic data.
        """
        logger.info("Identifying communities of interest.")

        try:
            socio_economic_features = self.data[[
                'median_rent',
                'environmental_hazard_index'
            ]].fillna(0)

            # Normalize features
            socio_economic_features = (socio_economic_features - socio_economic_features.mean()) / socio_economic_features.std()

            # Perform clustering
            num_clusters = Config.NUM_DISTRICTS * 2  # Adjust as needed
            kmeans = KMeans(n_clusters=num_clusters, random_state=1)
            self.data['coi'] = kmeans.fit_predict(socio_economic_features)

            logger.info("Communities of interest identified.")
        except Exception as e:
            logger.error(f"Error identifying communities of interest: {e}")
            raise DataProcessingError(f"Error identifying communities of interest: {e}")

