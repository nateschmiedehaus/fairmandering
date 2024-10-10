# fairmandering/config.py

import os
from dotenv import load_dotenv
import logging
from census import Census
from math import sqrt

load_dotenv()

class Config:
    """
    Configuration class for the redistricting system.
    """

    # REDIS configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

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

    # Genetic Algorithm Configuration
    GA_POPULATION_SIZE = int(os.getenv('GA_POPULATION_SIZE', '200'))
    GA_GENERATIONS = int(os.getenv('GA_GENERATIONS', '150'))
    GA_CROSSOVER_RATE = float(os.getenv('GA_CROSSOVER_RATE', '0.9'))
    GA_MUTATION_RATE = float(os.getenv('GA_MUTATION_RATE', '0.1'))
    GA_SELECTION_METHOD = os.getenv('GA_SELECTION_METHOD', 'tournament')

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
        try:
            response = cls.CENSUS.acs5.get(('B01001_001E',), {'for': f'state:{state_fips}'})
            state_population = int(response[0]['B01001_001E'])
            return cls.calculate_districts_from_population(state_population)
        except Exception as e:
            raise ValueError(f"Error retrieving district data: {e}")

    @staticmethod
    def calculate_districts_from_population(population: int) -> int:
        base_seats = 1
        national_population = 331002651
        standard_divisor = national_population / 435

        def huntington_hill_rounding(n, pop, divisor):
            return (n + 1) if pop / divisor > sqrt(n * (n + 1)) else n

        num_seats = base_seats
        while huntington_hill_rounding(num_seats, population, standard_divisor) > num_seats:
            num_seats += 1

        return num_seats

    # Paths
    SHAPEFILE_PATH = os.getenv('SHAPEFILE_PATH', f'shapefiles/{STATE_FIPS}/tl_2020_{STATE_FIPS}_tabblock20.shp')

    # Caching and Parallelization
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True').lower() in ['true', '1', 'yes']
    CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
    ENABLE_PARALLELIZATION = os.getenv('ENABLE_PARALLELIZATION', 'True').lower() in ['true', '1', 'yes']
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))

    # Security settings
    ENABLE_ENCRYPTION = os.getenv('ENABLE_ENCRYPTION', 'True').lower() in ['true', '1', 'yes']
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', None)

    # Flask Configuration
    FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']

    @classmethod
    def validate(cls) -> bool:
        required_keys = ['CENSUS_API_KEY', 'FEC_API_KEY', 'TABLEAU_SERVER_URL', 'ENCRYPTION_KEY']
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

        if cls.GA_POPULATION_SIZE <= 0:
            raise ValueError("GA_POPULATION_SIZE must be a positive integer.")
        if cls.GA_GENERATIONS <= 0:
            raise ValueError("GA_GENERATIONS must be a positive integer.")
        if not (0 <= cls.GA_CROSSOVER_RATE <= 1):
            raise ValueError("GA_CROSSOVER_RATE must be between 0 and 1.")
        if not (0 <= cls.GA_MUTATION_RATE <= 1):
            raise ValueError("GA_MUTATION_RATE must be between 0 and 1.")

        if not os.path.exists(cls.SHAPEFILE_PATH):
            logging.warning(f"Shapefile path does not exist: {cls.SHAPEFILE_PATH}. It will be downloaded.")

        if cls.ENABLE_CACHING and not os.path.exists(cls.CACHE_DIR):
            os.makedirs(cls.CACHE_DIR)

        if cls.ENABLE_PARALLELIZATION and cls.NUM_WORKERS <= 0:
            raise ValueError("NUM_WORKERS must be a positive integer.")

        return True

Config.validate()
