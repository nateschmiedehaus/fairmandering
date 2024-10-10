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
    visualize_trend_analysis
)
from .analysis import analyze_districts, save_analysis_results, perform_sensitivity_analysis, compare_ensemble_plans, rank_plans
from .versioning import save_plan
import os
import sys

app = Flask(__name__)
app.secret_key = Config.ENCRYPTION_KEY or 'default_secret_key'  # Ensure to set a secure key in .env

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
    Config.STATE_NAME = Config.STATE_NAME  # Optionally, map FIPS to state name

    # System Checks
    try:
        Config.validate()
        logger.info("Configuration validated successfully.")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        flash(f"Configuration validation failed: {e}", 'danger')
        return redirect(url_for('home'))

    # Data Processing
    processor = DataProcessor(state_fips, Config.STATE_NAME)
    try:
        data = processor.integrate_data()
    except DataProcessingError as e:
        logger.error(f"Data processing failed: {e}")
        flash(f"Data processing failed: {e}", 'danger')
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"Unexpected error during data processing: {e}")
        flash(f"Unexpected error during data processing: {e}", 'danger')
        return redirect(url_for('home'))

    # Optimization
    try:
        district_assignments, fitness_scores = optimize_districting(data, seeds=[42])  # Using a fixed seed for reproducibility
        best_assignment = district_assignments[0]  # Select the first solution as the best
        logger.info("Optimization completed successfully.")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        flash(f"Optimization failed: {e}", 'danger')
        return redirect(url_for('home'))

    # Fairness Evaluation
    try:
        fairness_metrics = evaluate_fairness(data, best_assignment)
        logger.info("Fairness evaluation completed.")
    except Exception as e:
        logger.error(f"Fairness evaluation failed: {e}")
        flash(f"Fairness evaluation failed: {e}", 'danger')
        return redirect(url_for('home'))

    # Analysis
    try:
        analysis_results = analyze_districts(data)
        save_analysis_results(analysis_results)
        logger.info("District analysis completed and results saved.")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        flash(f"Analysis failed: {e}", 'danger')
        return redirect(url_for('home'))

    # Visualization
    try:
        visualize_district_map(data, best_assignment)
        plot_fairness_metrics(fairness_metrics)
        visualize_district_characteristics(data)
        visualize_trend_analysis(data)
        generate_explainable_report(fairness_metrics, analysis_results)
        logger.info("Visualizations generated and saved.")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        flash(f"Visualization failed: {e}", 'danger')
        return redirect(url_for('home'))

    # Versioning
    try:
        metadata = {'author': 'Your Name', 'description': 'Initial plan'}
        plan_path = save_plan(best_assignment, metadata, version='1.0.0')
        logger.info(f"Plan versioned and saved at {plan_path}.")
    except Exception as e:
        logger.error(f"Versioning failed: {e}")
        flash(f"Versioning failed: {e}", 'danger')
        return redirect(url_for('home'))

    # Sensitivity Analysis
    try:
        perform_sensitivity_analysis(data, best_assignment)
        logger.info("Sensitivity analysis completed.")
    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}")
        flash(f"Sensitivity analysis failed: {e}", 'danger')
        return redirect(url_for('home'))

    # Ensemble Analysis
    try:
        ensemble = generate_ensemble_plans(data, num_plans=5)
        metrics_df = compare_ensemble_plans(data, ensemble)
        weights = Config.OBJECTIVE_WEIGHTS
        ranked_plans = rank_plans(metrics_df, weights)
        logger.info("Ensemble analysis completed.")
    except Exception as e:
        logger.error(f"Ensemble analysis failed: {e}")
        flash(f"Ensemble analysis failed: {e}", 'danger')
        return redirect(url_for('home'))

    flash("Redistricting process completed successfully!", 'success')
    return render_template('results.html',
                           fairness_metrics=fairness_metrics,
                           analysis_results=analysis_results,
                           ranked_plans=ranked_plans)

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
