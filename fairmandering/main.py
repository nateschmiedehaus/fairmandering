# fairmandering/main.py

import logging
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from .config import Config
from .data_processing import DataProcessor, DataProcessingError
from .optimization import optimize_districting, generate_ensemble_plans
from .fairness_evaluation import evaluate_fairness
from .visualization import (
    visualize_district_map,
    plot_fairness_metrics,
    visualize_district_characteristics,
    generate_explainable_report,
    visualize_trend_analysis,
    generate_comparative_analysis_plot
)
from .analysis import analyze_districts, save_analysis_results, perform_sensitivity_analysis, compare_ensemble_plans, rank_plans
from .versioning import save_plan
import os
from typing import List, Dict

app = Flask(__name__)
app.secret_key = Config.ENCRYPTION_KEY

# Configure logging
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=Config.LOG_LEVEL,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    """
    Home page of the Fairmandering GUI.
    """
    return render_template('home.html')

@app.route('/run', methods=['POST'])
def run_redistricting():
    """
    Handles the redistricting process initiated from the GUI.
    """
    state_fips = request.form.get('state_fips', Config.STATE_FIPS)
    logger.info(f"Redistricting process started for state FIPS: {state_fips}")

    # Update configuration if a different state is selected
    Config.STATE_FIPS = state_fips

    try:
        # System Checks
        Config.validate()
        logger.info("Configuration validated successfully.")
        
        # Data Processing
        processor = DataProcessor(state_fips, Config.STATE_NAME)
        data = processor.integrate_data()

        # Optimization
        district_assignments, fitness_scores = optimize_districting(data, seeds=[42])  # Using a fixed seed for reproducibility
        best_assignment = district_assignments[0]  # Select the first solution as the best
        logger.info("Optimization completed successfully.")

        # Fairness Evaluation
        fairness_metrics = evaluate_fairness(data, best_assignment)
        logger.info("Fairness evaluation completed.")

        # Analysis
        analysis_results = analyze_districts(data)
        save_analysis_results(analysis_results)
        logger.info("District analysis completed and results saved.")

        # Visualization
        visualization_paths = perform_visualizations(data, best_assignment, fairness_metrics)

        # Versioning
        version_plan(best_assignment, metadata={'author': 'Nathaniel Schmiedehaus', 'description': 'Initial plan'})

        # Sensitivity Analysis
        perform_sensitivity_analysis(data, best_assignment)
        logger.info("Sensitivity analysis completed.")

        flash("Redistricting process completed successfully!", 'success')
        return render_template('results.html',
                               fairness_metrics=fairness_metrics,
                               analysis_results=analysis_results,
                               comparative_plot_path=visualization_paths['comparative_plot_path'])

    except DataProcessingError as e:
        return handle_error("Data processing failed", e)
    except Exception as e:
        return handle_error("An unexpected error occurred", e)

def perform_visualizations(data, best_assignment, fairness_metrics):
    """
    Performs all visualizations and returns paths.

    Args:
        data: Integrated geospatial and demographic data.
        best_assignment: The best district assignment solution.
        fairness_metrics: Fairness metrics for evaluation.

    Returns:
        Dict containing paths to visualizations.
    """
    try:
        map_path = visualize_district_map(data, best_assignment)
        fairness_metrics_path = plot_fairness_metrics(fairness_metrics)
        characteristics_paths = visualize_district_characteristics(data)
        trend_path = visualize_trend_analysis(data)

        # Generate comparative analysis if ensemble plans are available
        ensemble = generate_ensemble_plans(data, num_plans=5)
        ensemble_metrics = [evaluate_fairness(data, assignment) for assignment in ensemble]
        comparative_plot_path = generate_comparative_analysis_plot(ensemble_metrics)
        logger.info("Comparative analysis generated.")

        report_path = generate_explainable_report(fairness_metrics, analysis_results)
        logger.info("Explainable report generated.")

        return {
            'map_path': map_path,
            'fairness_metrics_path': fairness_metrics_path,
            'characteristics_paths': characteristics_paths,
            'trend_path': trend_path,
            'comparative_plot_path': comparative_plot_path,
            'report_path': report_path
        }

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        flash(f"Visualization failed: {e}", 'danger')
        raise

def version_plan(best_assignment, metadata):
    """
    Versions the plan and saves it.

    Args:
        best_assignment: The best assignment of districts.
        metadata: Metadata for the plan.
    """
    try:
        plan_path = save_plan(best_assignment, metadata, version='1.0.0')
        logger.info(f"Plan versioned and saved at {plan_path}.")
    except Exception as e:
        logger.error(f"Versioning failed: {e}")
        flash(f"Versioning failed: {e}", 'danger')
        raise

def handle_error(message, exception):
    """
    Handles errors by logging and flashing messages.

    Args:
        message: Custom error message.
        exception: The exception raised.
    """
    logger.error(f"{message}: {exception}")
    flash(f"{message}: {exception}", 'danger')
    return redirect(url_for('home'))

@app.route('/tableau')
def tableau_dashboard():
    """
    Route for displaying Tableau Public dashboard.
    """
    return render_template('tableau_dashboard.html')

@app.route('/download/<filename>')
def download_file(filename):
    """
    Allows users to download generated files.
    """
    try:
        return send_file(os.path.join(os.getcwd(), filename), as_attachment=True)
    except Exception as e:
        logger.error(f"File download failed: {e}")
        flash(f"File download failed: {e}", 'danger')
        return redirect(url_for('home'))

def run_flask_app():
    """
    Runs the Flask web application.
    """
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)

if __name__ == "__main__":
    run_flask_app()
