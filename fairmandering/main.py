import logging
from .config import Config
from .data_processing import DataProcessor, DataProcessingError
from .optimization import optimize_districting, generate_ensemble_plans
from .fairness_evaluation import evaluate_fairness
from .visualization import (
    visualize_district_map,
    plot_fairness_metrics,
    visualize_district_characteristics,
    generate_explainable_report,
    visualize_trend_analysis
)
from .analysis import analyze_districts, save_analysis_results, perform_sensitivity_analysis, compare_ensemble_plans, rank_plans
from .versioning import save_plan
import argparse
import sys

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fairmandering Redistricting System')
    parser.add_argument('--state_fips', help='State FIPS code', default=Config.STATE_FIPS)
    return parser.parse_args()

def system_check():
    """
    Verifies all dependencies, API keys, and data sources before main execution.
    """
    logger.info("Performing system checks.")
    try:
        # Validate configuration
        Config.validate()
        logger.info("Configuration validated successfully.")

        # Check for required packages
        required_packages = [
            'pandas', 'geopandas', 'numpy', 'scipy', 'requests', 'census', 'pymoo',
            'matplotlib', 'seaborn', 'folium', 'python-dotenv', 'joblib', 'scikit-learn',
            'cryptography', 'us', 'plotly', 'redis'
        ]
        for pkg in required_packages:
            __import__(pkg)
        logger.info("All required packages are installed.")

    except Exception as e:
        logger.error(f"System check failed: {e}")
        sys.exit(1)

def main():
    logging.basicConfig(level=Config.LOG_LEVEL)
    logger.info("Starting the redistricting process.")

    # Perform system checks
    system_check()

    # Parse arguments
    args = parse_arguments()
    state_fips = args.state_fips

    # Get the number of districts dynamically from the Census API
    try:
        num_districts = Config.get_num_districts(state_fips)
    except Exception as e:
        logger.error(f"Failed to get the number of districts: {e}")
        sys.exit(1)

    # Data Processing
    processor = DataProcessor(state_fips, Config.STATE_NAME)
    try:
        data = processor.integrate_data()
    except DataProcessingError as e:
        logger.error(f"Data processing failed: {e}")
        sys.exit(1)

    # Optimization
    try:
        district_assignments, _ = optimize_d
