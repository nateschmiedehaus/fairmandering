# fairmandering/config.py

import os
from dotenv import load_dotenv
import logging
from census import Census
from math import sqrt

load_dotenv()

class Config:
    """
    Configuration class that holds all settings and parameters for the redistricting system.
    """

    # REDIS configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

    # TABLEAU Configuration
    TABLEAU_SERVER_URL = 'https://public.tableau.com/'


    # Logging configuration
    LOG_FILE = 'fairmandering.log'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # API Keys (ensure these are set in your .env file)
    CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
    FEC_API_KEY = os.getenv('FEC_API_KEY')
    BLS_API_KEY = os.getenv('BLS_API_KEY')
    HUD_API_KEY = os.getenv('HUD_API_KEY')
    EPA_API_KEY = os.getenv('EPA_API_KEY')

    # Census object initialization
    CENSUS = Census(CENSUS_API_KEY)

    # Optimization parameters for Genetic Algorithm
    GA_POPULATION_SIZE = int(os.getenv('GA_POPULATION_SIZE', '200'))
    GA_GENERATIONS = int(os.getenv('GA_GENERATIONS', '150'))
    GA_CROSSOVER_RATE = float(os.getenv('GA_CROSSOVER_RATE', '0.9'))
    GA_MUTATION_RATE = float(os.getenv('GA_MUTATION_RATE', '0.1'))
    GA_SELECTION_METHOD = os.getenv('GA_SELECTION_METHOD', 'tournament')  # Options: 'tournament', 'roulette', etc.

    # Adaptive weighting for objectives
    OBJECTIVE_WEIGHTS = {
        'population_equality': float(os.getenv('WEIGHT_POPULATION_EQUALITY', '1.0')),
        'compactness': float(os.getenv('WEIGHT_COMPACTNESS', '1.0')),
        'minority_representation': float(os.getenv('WEIGHT_MINOR_REPRESENTATION', '1.0')),
        'political_fairness': float(os.getenv('WEIGHT_POLITICAL_FAIRNESS', '1.0')),
        'competitiveness': float(os.getenv('WEIGHT_COMPETITIVENESS', '1.0')),
        'coi_preservation': float(os.getenv('WEIGHT_COI_PRESERVATION', '1.0')),
        'socioeconomic_parity': float(os.getenv('WEIGHT_SOCIOECONOMIC_PARITY', '1.0')),
        'environmental_justice': float(os.getenv('WEIGHT_ENVIRONMENTAL_JUSTICE', '1.0')),
        'trend_consideration': float(os.getenv('WEIGHT_TREND_CONSIDERATION', '1.0'))
    }

    # State configuration
    STATE_FIPS = os.getenv('STATE_FIPS', '06')  # Default to California
    STATE_NAME = os.getenv('STATE_NAME', 'California')

    @classmethod
    def get_num_districts(cls, state_fips: str) -> int:
        """
        Retrieves the number of legally required districts for a given state using Census data
        and the official apportionment rules.

        Args:
            state_fips (str): The FIPS code of the state.

        Returns:
            int: The calculated number of congressional districts.
        """
        try:
            # Retrieve the total population for the state using the Census API
            response = cls.CENSUS.acs5.get(
                ('B01001_001E',),  # B01001_001E is the total population estimate variable
                {'for': f'state:{state_fips}'}
            )
            state_population = int(response[0]['B01001_001E'])

            # Calculate the number of districts using the Huntington-Hill method
            num_districts = cls.calculate_districts_from_population(state_population)
            return num_districts

        except Exception as e:
            raise ValueError(f"Error retrieving district data: {e}")

    @staticmethod
    def calculate_districts_from_population(population: int) -> int:
        """
        Calculates the number of congressional districts based on the state's population using
        the Huntington-Hill method. This method is consistent with how the U.S. House of Representatives
        is apportioned.

        Args:
            population (int): The population of the state.

        Returns:
            int: The calculated number of congressional districts.
        """
        base_seats = 1
        national_population = 331002651  # Example: 2020 U.S. population
        standard_divisor = national_population / 435

        def huntington_hill_rounding(n, pop, divisor):
            return (n + 1) if pop / divisor > sqrt(n * (n + 1)) else n

        num_seats = base_seats
        while huntington_hill_rounding(num_seats, population, standard_divisor) > num_seats:
            num_seats += 1

        return num_seats

    # Paths
    SHAPEFILE_PATH = os.getenv(
        'SHAPEFILE_PATH',
        f'shapefiles/{STATE_FIPS}/tl_2020_{STATE_FIPS}_tabblock20.shp'
    )

    # Caching settings
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True') == 'True'
    CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
    CACHE_EXPIRATION_TIME = int(os.getenv('CACHE_EXPIRATION_TIME', '86400'))  # In seconds

    # Parallelization settings
    ENABLE_PARALLELIZATION = os.getenv('ENABLE_PARALLELIZATION', 'True') == 'True'
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))

    # Trend analysis settings
    TREND_YEARS = [int(year) for year in os.getenv('TREND_YEARS', '2000,2010,2020').split(',')]

    # Security settings
    ENABLE_ENCRYPTION = os.getenv('ENABLE_ENCRYPTION', 'True') == 'True'
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', None)  # No default value for security

    # Flask Configuration for GUI
    FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'

    @classmethod
    def validate(cls) -> bool:
        """
        Validates the configuration parameters to ensure they are set correctly.

        Returns:
            bool: True if all validations pass.

        Raises:
            ValueError: If any configuration parameter is invalid or missing.
        """
        api_keys = {
            'CENSUS_API_KEY': cls.CENSUS_API_KEY,
            'FEC_API_KEY': cls.FEC_API_KEY,
            'BLS_API_KEY': cls.BLS_API_KEY,
            'HUD_API_KEY': cls.HUD_API_KEY,
            'EPA_API_KEY': cls.EPA_API_KEY
        }
        for key, value in api_keys.items():
            if not value:
                raise ValueError(f"Missing API key: {key}. Please set it in the .env file.")

        if cls.GA_POPULATION_SIZE <= 0:
            raise ValueError("GA_POPULATION_SIZE must be a positive integer.")
        if cls.GA_GENERATIONS <= 0:
            raise ValueError("GA_GENERATIONS must be a positive integer.")
        if not (0 <= cls.GA_CROSSOVER_RATE <= 1):
            raise ValueError("GA_CROSSOVER_RATE must be between 0 and 1.")
        if not (0 <= cls.GA_MUTATION_RATE <= 1):
            raise ValueError("GA_MUTATION_RATE must be between 0 and 1.")

        for weight in cls.OBJECTIVE_WEIGHTS.values():
            if weight < 0:
                raise ValueError("Objective weights must be non-negative.")

        if not os.path.exists(cls.SHAPEFILE_PATH):
            logging.warning(f"Shapefile path does not exist: {cls.SHAPEFILE_PATH}. It will be downloaded.")

        if cls.ENABLE_CACHING and not os.path.exists(cls.CACHE_DIR):
            os.makedirs(cls.CACHE_DIR)

        if cls.ENABLE_PARALLELIZATION and cls.NUM_WORKERS <= 0:
            raise ValueError("NUM_WORKERS must be a positive integer.")

        if not cls.TREND_YEARS or not all(isinstance(year, int) for year in cls.TREND_YEARS):
            raise ValueError("TREND_YEARS must be a list of integers representing years.")

        if cls.ENABLE_ENCRYPTION and not cls.ENCRYPTION_KEY:
            raise ValueError("ENCRYPTION_KEY must be set if encryption is enabled.")

        return True

    @classmethod
    def validate(cls):
        required_vars = ['TABLEAU_SERVER_URL', 'FLASK_SECRET_KEY']
        for var in required_vars:
            if not getattr(cls, var):
                raise ValueError(f"Missing required environment variable: {var}")

Config.validate()
