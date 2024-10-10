# fairmandering/analysis.py

import pandas as pd
import logging
import matplotlib.pyplot as plt
from .config import Config
from .optimization import optimize_districting
from .fairness_evaluation import evaluate_fairness

logger = logging.getLogger(__name__)

def analyze_districts(data):
    """
    Performs detailed analysis of each district.

    Args:
        data (GeoDataFrame): The geospatial data with district assignments.

    Returns:
        dict: Analysis results containing various statistics per district.
    """
    logger.info("Analyzing districts.")

    analysis_results = {}
    num_districts = Config.NUM_DISTRICTS

    for i in range(num_districts):
        district_data = data[data['district'] == i]
        total_population = district_data['P001001'].sum()
        minority_population = district_data['P005004'].sum()
        median_income = district_data['median_rent'].median()
        environmental_hazard = district_data['environmental_hazard_index'].mean()
        competitiveness = abs(
            district_data['votes_party_a'].sum() - district_data['votes_party_b'].sum()
        )
        # Additional analysis metrics
        area = district_data.geometry.area.sum()
        compactness = area / (district_data.geometry.length.sum() ** 2)

        analysis_results[i] = {
            'total_population': total_population,
            'minority_population': minority_population,
            'median_income': median_income,
            'environmental_hazard_index': environmental_hazard,
            'competitiveness': competitiveness,
            'area': area,
            'compactness': compactness,
        }

    logger.info("District analysis completed.")
    return analysis_results

def save_analysis_results(analysis_results):
    """
    Saves the analysis results to a CSV file.

    Args:
        analysis_results (dict): The analysis results to save.
    """
    logger.info("Saving analysis results.")
    df = pd.DataFrame.from_dict(analysis_results, orient='index')
    df.to_csv('district_analysis.csv')
    logger.info("Analysis results saved as 'district_analysis.csv'.")

def perform_sensitivity_analysis(data, assignment):
    """
    Performs sensitivity analysis on the redistricting plan.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.
        assignment (array): Array assigning each unit to a district.
    """
    logger.info("Performing sensitivity analysis.")

    original_weight = Config.OBJECTIVE_WEIGHTS['population_equality']
    deviations = []
    weights = [0.5 * original_weight, original_weight, 1.5 * original_weight]

    for weight in weights:
        Config.OBJECTIVE_WEIGHTS['population_equality'] = weight
        assignments, _ = optimize_districting(data)
        best_assignment = assignments[0]
        fairness_metrics = evaluate_fairness(data, best_assignment)
        deviations.append(fairness_metrics['population_equality'])

    Config.OBJECTIVE_WEIGHTS['population_equality'] = original_weight

    # Plotting the sensitivity analysis
    plt.figure(figsize=(10, 6))
    plt.plot(weights, deviations, marker='o')
    plt.xlabel('Population Equality Weight')
    plt.ylabel('Population Deviation')
    plt.title('Sensitivity Analysis of Population Equality Weight')
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png')
    plt.close()
    logger.info("Sensitivity analysis completed and saved as 'sensitivity_analysis.png'.")

def compare_ensemble_plans(data, ensemble):
    """
    Compares and ranks ensemble of redistricting plans.

    Args:
        data (GeoDataFrame): The geospatial data.
        ensemble (list): List of district assignments.

    Returns:
        DataFrame: DataFrame containing metrics for each plan.
    """
    logger.info("Comparing ensemble of plans.")
    metrics = []
    for assignment in ensemble:
        fairness_metrics = evaluate_fairness(data, assignment)
        metrics.append(fairness_metrics)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('ensemble_metrics.csv')
    logger.info("Ensemble metrics saved as 'ensemble_metrics.csv'.")
    return metrics_df

def rank_plans(metrics_df, weights):
    """
    Ranks plans based on weighted criteria.

    Args:
        metrics_df (DataFrame): DataFrame containing metrics for each plan.
        weights (dict): Weights for each metric.

    Returns:
        DataFrame: Ranked plans.
    """
    logger.info("Ranking plans based on weighted criteria.")
    for metric in weights:
        if metric in metrics_df.columns:
            metrics_df[metric] = metrics_df[metric] * weights[metric]
    metrics_df['total_score'] = metrics_df.sum(axis=1)
    ranked_plans = metrics_df.sort_values('total_score')
    ranked_plans.to_csv('ranked_plans.csv')
    logger.info("Plans ranked and saved as 'ranked_plans.csv'.")
    return ranked_plans
