#config.py
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

    # Optimization parameters
    NSGA3_POPULATION_SIZE = int(os.getenv('NSGA3_POPULATION_SIZE', '200'))
    NSGA3_GENERATIONS = int(os.getenv('NSGA3_GENERATIONS', '150'))
    STOCHASTIC_MUTATION_PROB = float(os.getenv('STOCHASTIC_MUTATION_PROB', '0.1'))

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
    def get_num_districts(cls, state_fips):
        """
        Retrieves the number of legally required districts for a given state using Census data
        and the official apportionment rules.
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
    def calculate_districts_from_population(population):
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
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'your_encryption_key_here')

    @classmethod
    def validate(cls):
        """
        Validates the configuration parameters to ensure they are set correctly.
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

        if cls.NSGA3_POPULATION_SIZE <= 0:
            raise ValueError("NSGA3_POPULATION_SIZE must be a positive integer.")
        if cls.NSGA3_GENERATIONS <= 0:
            raise ValueError("NSGA3_GENERATIONS must be a positive integer.")
        if not (0 <= cls.STOCHASTIC_MUTATION_PROB <= 1):
            raise ValueError("STOCHASTIC_MUTATION_PROB must be between 0 and 1.")

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
