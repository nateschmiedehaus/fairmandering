# fairmandering/optimization.py

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from typing import Tuple, List
import logging
from .fairness_evaluation import evaluate_fairness
from .config import Config

logger = logging.getLogger(__name__)

class DistrictingProblem(Problem):
    """
    Defines the optimization problem for redistricting using pymoo's Problem class.
    """

    def __init__(self, data: 'GeoDataFrame', num_districts: int):
        """
        Initializes the DistrictingProblem.

        Args:
            data (GeoDataFrame): The geospatial data with demographic and other attributes.
            num_districts (int): The number of districts to create.
        """
        self.data = data
        self.num_districts = num_districts
        num_units = len(data)
        super().__init__(n_var=num_units,
                         n_obj=len(Config.OBJECTIVE_WEIGHTS),
                         n_constr=0,
                         xl=0,
                         xu=num_districts - 1,
                         type_var=int)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluates the objective functions for a batch of solutions.

        Args:
            X (ndarray): Solution matrix where each row represents a solution.
            out (dict): Dictionary to store the objective function values.
        """
        objectives = []
        for solution in X:
            assignment = solution.astype(int)
            fairness_metrics = evaluate_fairness(self.data, assignment)
            # Objective function: Weighted sum of metrics
            weighted_metrics = [
                Config.OBJECTIVE_WEIGHTS['population_equality'] * fairness_metrics.get('population_equality', 0),
                Config.OBJECTIVE_WEIGHTS['compactness'] * self.calculate_compactness(assignment),
                Config.OBJECTIVE_WEIGHTS['minority_representation'] * fairness_metrics.get('minority_representation', 0),
                Config.OBJECTIVE_WEIGHTS['political_fairness'] * fairness_metrics.get('political_fairness', 0),
                Config.OBJECTIVE_WEIGHTS['competitiveness'] * fairness_metrics.get('competitiveness', 0),
                # Add other weighted metrics as needed
            ]
            objectives.append(weighted_metrics)
        out["F"] = np.array(objectives)

    def calculate_compactness(self, assignment: np.ndarray) -> float:
        """
        Calculates the compactness of the districting plan using Polsby-Popper score.

        Args:
            assignment (np.ndarray): Array assigning each unit to a district.

        Returns:
            float: Average compactness score across all districts.
        """
        compactness_scores = []
        for district in range(self.num_districts):
            district_units = self.data[assignment == district]
            if district_units.empty:
                compactness_scores.append(0)
                continue
            perimeter = district_units.geometry.length.sum()
            area = district_units.geometry.area.sum()
            if perimeter == 0:
                compactness_scores.append(0)
                continue
            polsby_popper = (4 * np.pi * area) / (perimeter ** 2)
            compactness_scores.append(polsby_popper)
        average_compactness = np.mean(compactness_scores) if compactness_scores else 0
        return average_compactness

def optimize_districting(data: 'GeoDataFrame', seeds: List[int] = None) -> Tuple[List[np.ndarray], List[float]]:
    """
    Optimizes the districting plan using a genetic algorithm.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.
        seeds (List[int], optional): List of random seeds for reproducibility.

    Returns:
        Tuple[List[np.ndarray], List[float]]: A tuple containing the list of district assignments and their corresponding fitness scores.
    """
    num_districts = Config.get_num_districts(Config.STATE_FIPS)
    problem = DistrictingProblem(data, num_districts)

    reference_directions = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)

    algorithm = NSGA3(pop_size=Config.GA_POPULATION_SIZE,
                      ref_dirs=reference_directions)

    termination = get_termination("n_gen", Config.GA_GENERATIONS)

    logger.info("Starting optimization using NSGA-III.")
    try:
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=seeds[0] if seeds else None,
                       save_history=True,
                       verbose=True)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

    assignments = res.X.astype(int).tolist()
    fitness_scores = res.F.tolist()

    logger.info("Optimization completed successfully.")
    return assignments, fitness_scores

def generate_ensemble_plans(data: 'GeoDataFrame', num_plans: int = 5) -> List[np.ndarray]:
    """
    Generates an ensemble of redistricting plans using different seeds.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.
        num_plans (int, optional): Number of plans to generate.

    Returns:
        List[np.ndarray]: List of district assignments for each plan.
    """
    ensemble = []
    for seed in range(num_plans):
        logger.info(f"Generating ensemble plan {seed + 1} with seed {seed}.")
        assignments, _ = optimize_districting(data, seeds=[seed])
        ensemble.append(np.array(assignments))
    return ensemble
