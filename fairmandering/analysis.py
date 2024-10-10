# fairmandering/analysis.py

import pandas as pd
import logging
from typing import Dict, Any
from .config import Config

logger = logging.getLogger(__name__)

def analyze_districts(data: 'GeoDataFrame') -> Dict[int, Dict[str, Any]]:
    """
    Analyzes each district to calculate various metrics.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary with district numbers as keys and their metrics as values.
    """
    logger.info("Analyzing districts.")
    analysis_results = {}
    num_districts = len(pd.unique(data['district']))

    for district in range(num_districts):
        district_data = data[data['district'] == district]
        population = district_data['P001001'].sum()
        minority_population = district_data['P005004'].sum()
        median_income = district_data['median_income'].median()
        # Add more metrics as needed

        analysis_results[district] = {
            'population': population,
            'minority_population': minority_population,
            'median_income': median_income,
            # Add more metrics here
        }

    logger.info("District analysis completed.")
    return analysis_results

def save_analysis_results(analysis_results: Dict[int, Dict[str, Any]]) -> None:
    """
    Saves the analysis results to a CSV file.

    Args:
        analysis_results (Dict[int, Dict[str, Any]]): The analysis results to save.
    """
    logger.info("Saving analysis results.")
    df = pd.DataFrame.from_dict(analysis_results, orient='index')
    df.index.name = 'District'
    df.to_csv('static/district_analysis.csv')
    logger.info("Analysis results saved as 'static/district_analysis.csv'.")

def perform_sensitivity_analysis(data: 'GeoDataFrame', assignment: np.ndarray) -> None:
    """
    Performs sensitivity analysis on the redistricting plan.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.
        assignment (np.ndarray): Array assigning each unit to a district.
    """
    logger.info("Performing sensitivity analysis.")

    original_weight = Config.OBJECTIVE_WEIGHTS.get('population_equality', 1.0)
    deviations = []
    weights = [0.5 * original_weight, original_weight, 1.5 * original_weight]

    for weight in weights:
        Config.OBJECTIVE_WEIGHTS['population_equality'] = weight
        logger.info(f"Adjusting population equality weight to {weight}.")
        # Re-run optimization with adjusted weight
        processor = DataProcessor(Config.STATE_FIPS, Config.STATE_NAME)
        processed_data = processor.integrate_data()  # Fetch processed and integrated data
        assignments, _ = optimize_districting(processed_data)
        best_assignment = assignments[0]
        fairness_metrics = evaluate_fairness(processed_data, best_assignment)
        deviations.append(fairness_metrics.get('population_equality', 0))

    Config.OBJECTIVE_WEIGHTS['population_equality'] = original_weight
    logger.info(f"Reset population equality weight to original value {original_weight}.")

    # Create Sensitivity Analysis Plot
    import plotly.express as px

    fig = px.line(
        x=weights,
        y=deviations,
        labels={'x': 'Population Equality Weight', 'y': 'Population Deviation'},
        title='Sensitivity Analysis of Population Equality Weight',
        markers=True
    )
    fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
    sensitivity_filename = 'static/sensitivity_analysis.html'
    fig.write_html(sensitivity_filename, full_html=False, include_plotlyjs='cdn')
    logger.info(f"Sensitivity analysis plot saved as '{sensitivity_filename}'.")

def compare_ensemble_plans(data: 'GeoDataFrame', ensemble: List[np.ndarray]) -> pd.DataFrame:
    """
    Compares and ranks ensemble of redistricting plans.

    Args:
        data (GeoDataFrame): The geospatial data.
        ensemble (List[np.ndarray]): List of district assignments.

    Returns:
        pd.DataFrame: DataFrame containing metrics for each plan.
    """
    logger.info("Comparing ensemble of plans.")
    metrics = []
    for i, assignment in enumerate(ensemble):
        fairness_metrics = evaluate_fairness(data, assignment)
        metrics.append(fairness_metrics)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.index = [f'Plan {i+1}' for i in range(len(ensemble))]
    metrics_df.to_csv('static/ensemble_metrics.csv')
    logger.info("Ensemble metrics saved as 'static/ensemble_metrics.csv'.")
    return metrics_df

def rank_plans(metrics_df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Ranks plans based on weighted criteria.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics for each plan.
        weights (Dict[str, float]): Weights for each metric.

    Returns:
        pd.DataFrame: Ranked plans.
    """
    logger.info("Ranking plans based on weighted criteria.")
    # Normalize metrics if necessary
    for metric in weights:
        if metric in metrics_df.columns:
            max_val = metrics_df[metric].max()
            if max_val > 0:
                metrics_df[f"{metric}_normalized"] = metrics_df[metric] / max_val
            else:
                metrics_df[f"{metric}_normalized"] = 0

    # Apply weights
    for metric, weight in weights.items():
        if f"{metric}_normalized" in metrics_df.columns:
            metrics_df[metric] = metrics_df[f"{metric}_normalized"] * weight

    # Calculate total score
    weighted_columns = [metric for metric in weights if metric in metrics_df.columns]
    metrics_df['total_score'] = metrics_df[weighted_columns].sum(axis=1)

    # Sort by total score
    ranked_plans = metrics_df.sort_values('total_score')
    ranked_plans.to_csv('static/ranked_plans.csv')
    logger.info("Plans ranked and saved as 'static/ranked_plans.csv'.")
    return ranked_plans
