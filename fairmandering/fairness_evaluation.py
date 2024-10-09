# fairmandering/fairness_evaluation.py

import numpy as np
import logging
from .config import Config

logger = logging.getLogger(__name__)

def evaluate_fairness(data, assignment):
    """
    Evaluates the fairness of a redistricting plan.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.
        assignment (array): Array assigning each unit to a district.

    Returns:
        dict: Fairness metrics.
    """
    logger.info("Evaluating fairness of the districting plan.")

    num_districts = Config.NUM_DISTRICTS
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

    # Political Fairness
    party_a_votes = np.array([
        data.loc[assignment == i, 'votes_party_a'].sum() for i in range(num_districts)
    ])
    party_b_votes = np.array([
        data.loc[assignment == i, 'votes_party_b'].sum() for i in range(num_districts)
    ])
    total_votes = party_a_votes + party_b_votes
    vote_shares = party_a_votes / total_votes
    deviation = np.abs(vote_shares - 0.5)
    fairness_metrics['political_fairness'] = deviation.mean()

    # Competitiveness
    margins = np.abs(party_a_votes - party_b_votes)
    competitiveness = margins / total_votes
    fairness_metrics['competitiveness'] = competitiveness.mean()

    logger.info("Fairness evaluation completed.")
    return fairness_metrics
