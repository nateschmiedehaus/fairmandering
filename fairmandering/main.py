#main.py

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
        logger.info(f"Number of districts for state FIPS {state_fips}: {num_districts}")
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
        district_assignments, _ = optimize_districting(data, seeds=[1, 2, 3, 4, 5])
        best_assignment = district_assignments[0]  # For simplicity, use the first solution
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

    # Fairness Evaluation
    try:
        fairness_metrics = evaluate_fairness(data, best_assignment)
    except Exception as e:
        logger.error(f"Fairness evaluation failed: {e}")
        sys.exit(1)

    # Analysis
    try:
        analysis_results = analyze_districts(data)
        save_analysis_results(analysis_results)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

    # Visualization
    try:
        visualize_district_map(data, best_assignment)
        plot_fairness_metrics(fairness_metrics)
        visualize_district_characteristics(data)
        visualize_trend_analysis(data)
        generate_explainable_report(fairness_metrics, analysis_results)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)

    # Versioning
    metadata = {'author': 'Your Name', 'description': 'Initial plan'}
    save_plan(best_assignment, metadata, version='1.0.0')

    # Sensitivity Analysis
    perform_sensitivity_analysis(data, best_assignment)

    # Ensemble Analysis
    ensemble = generate_ensemble_plans(data, num_plans=5)
    metrics_df = compare_ensemble_plans(data, ensemble)
    weights = Config.OBJECTIVE_WEIGHTS
    ranked_plans = rank_plans(metrics_df, weights)

    logger.info("Redistricting process completed successfully.")

if __name__ == "__main__":
    args = parse_arguments()
    main()
