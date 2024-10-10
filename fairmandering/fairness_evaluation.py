# fairmandering/fairness_evaluation.py

import numpy as np
import logging
from typing import Dict
from .config import Config

logger = logging.getLogger(__name__)

def evaluate_fairness(data: 'GeoDataFrame', assignment: np.ndarray) -> Dict[str, float]:
    """
    Evaluates the fairness of a redistricting plan.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.
        assignment (np.ndarray): Array assigning each unit to a district.

    Returns:
        Dict[str, float]: Dictionary containing fairness metrics.
    """
    logger.info("Evaluating fairness of the districting plan.")
    num_districts = len(np.unique(assignment))
    fairness_metrics = {}

    # Population Equality
    population = data['P001001'].values
    total_population = population.sum()
    ideal_population = total_population / num_districts
    district_populations = np.array([
        population[assignment == i].sum() for i in range(num_districts)
    ])
    deviations = np.abs(district_populations - ideal_population) / ideal_population
    fairness_metrics['population_equality'] = deviations.max()

    # Minority Representation
    minority_populations = np.array([
        data.loc[assignment == i, 'P005004'].sum() for i in range(num_districts)
    ])
    total_minority_population = data['P005004'].sum()
    representation = minority_populations / total_minority_population
    fairness_metrics['minority_representation'] = representation.mean()

    # Political Fairness (Historical Trends Included)
    party_a_votes = np.array([
        data.loc[assignment == i, 'votes_party_a'].sum() for i in range(num_districts)
    ])
    party_b_votes = np.array([
        data.loc[assignment == i, 'votes_party_b'].sum() for i in range(num_districts)
    ])
    total_votes = party_a_votes + party_b_votes
    vote_shares = np.where(total_votes != 0, party_a_votes / total_votes, 0.5)  # Handle zero division cases
    deviation = np.abs(vote_shares - 0.5)
    fairness_metrics['political_fairness'] = deviation.mean()

    # Competitiveness
    margins = np.abs(party_a_votes - party_b_votes)
    competitiveness = np.where(total_votes != 0, margins / total_votes, 1.0)  # Handle zero division cases
    fairness_metrics['competitiveness'] = competitiveness.mean()

    # Compactness (Polsby-Popper and Additional Metric)
    compactness_scores = []
    convex_hull_scores = []
    for district in range(num_districts):
        district_units = data[assignment == district]
        if district_units.empty:
            compactness_scores.append(0)
            convex_hull_scores.append(0)
            continue
        perimeter = district_units.geometry.length.sum()
        area = district_units.geometry.area.sum()
        if perimeter == 0 or area == 0:
            compactness_scores.append(0)
            convex_hull_scores.append(0)
            continue
        polsby_popper = (4 * np.pi * area) / (perimeter ** 2)
        compactness_scores.append(polsby_popper)

        # Additional Compactness: Convex Hull
        convex_hull = district_units.unary_union.convex_hull
        convex_hull_area = convex_hull.area
        convex_hull_score = area / convex_hull_area
        convex_hull_scores.append(convex_hull_score)

    fairness_metrics['compactness_polsby_popper'] = np.mean(compactness_scores)
    fairness_metrics['compactness_convex_hull'] = np.mean(convex_hull_scores)

    # Socioeconomic Parity (Expanded Metrics including HUD and BLS)
    median_income = np.array([
        data.loc[assignment == i, 'B19013_001E'].median() for i in range(num_districts)
    ])
    unemployment_rate = np.array([
        data.loc[assignment == i, 'B23025_003E'].mean() for i in range(num_districts)
    ])
    median_home_value = np.array([
        data.loc[assignment == i, 'B25077_001E'].median() for i in range(num_districts)
    ])
    socioeconomic_parity = (np.std(median_income) / np.mean(median_income)) + \
                           (np.std(unemployment_rate) / np.mean(unemployment_rate)) + \
                           (np.std(median_home_value) / np.mean(median_home_value))
    fairness_metrics['socioeconomic_parity'] = socioeconomic_parity

    # Environmental Justice (if EPA data is available)
    if 'pollution_level' in data.columns:
        pollution_levels = np.array([
            data.loc[assignment == i, 'pollution_level'].mean() for i in range(num_districts)
        ])
        fairness_metrics['environmental_justice'] = np.std(pollution_levels) / np.mean(pollution_levels)

    # Stability and Growth Trends (Using trend analysis data)
    population_trend_mean = np.array([
        data.loc[assignment == i, 'population_trend_mean'].mean() for i in range(num_districts)
    ])
    income_trend_mean = np.array([
        data.loc[assignment == i, 'income_trend_mean'].mean() for i in range(num_districts)
    ])
    fairness_metrics['population_stability'] = np.std(population_trend_mean) / np.mean(population_trend_mean)
    fairness_metrics['income_stability'] = np.std(income_trend_mean) / np.mean(income_trend_mean)

    logger.info("Fairness evaluation completed.")
    return fairness_metrics
