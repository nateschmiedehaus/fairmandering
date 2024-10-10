# fairmandering/optimization.py

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.mutation import PolynomialMutation
from pymoo.operators.sampling import IntegerRandomSampling
from pymoo.operators.crossover import SBX
from pymoo.optimize import minimize
import logging
from .config import Config
from sklearn.neighbors import kneighbors_graph
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

class RedistrictingProblem(Problem):
    """
    Defines the multi-objective optimization problem for fair redistricting.
    """

    def __init__(self, data):
        num_districts = Config.get_dynamic_district_count(data)  # Use dynamic district count method
        super().__init__(n_var=len(data), n_obj=9, n_constr=1, type_var=int)
        self.data = data.reset_index(drop=True)
        self.num_districts = num_districts
        self.weights = Config.OBJECTIVE_WEIGHTS
        self.adjacency_matrix = self.build_adjacency_matrix()
        self.population = self.data['P001001'].values
        self.total_population = self.population.sum()
        self.ideal_population = self.total_population / self.num_districts

    def build_adjacency_matrix(self):
        """
        Builds an adjacency matrix based on spatial relationships.
        """
        logger.info("Building adjacency matrix.")
        adjacency = kneighbors_graph(self.data.geometry.apply(lambda geom: geom.centroid.coords[0]).tolist(),
                                     n_neighbors=8, mode='connectivity', include_self=False)
        adjacency = adjacency + adjacency.T
        return adjacency

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluates the objective functions and constraints for the population.
        """
        n_samples = X.shape[0]
        F = np.zeros((n_samples, self.n_obj))
        G = np.zeros((n_samples, self.n_constr))

        if Config.ENABLE_PARALLELIZATION:
            results = Parallel(n_jobs=Config.NUM_WORKERS)(
                delayed(self.evaluate_individual)(X[i, :]) for i in range(n_samples)
            )
        else:
            results = [self.evaluate_individual(X[i, :]) for i in range(n_samples)]

        for i, (f_vals, g_vals) in enumerate(results):
            F[i, :] = f_vals
            G[i, :] = g_vals

        out["F"] = F
        out["G"] = G

    def evaluate_individual(self, assignment):
        """
        Evaluates a single individual's objectives and constraints.
        """
        f = np.zeros(self.n_obj)
        g = np.zeros(self.n_constr)

        # Objectives
        f[0] = self.population_equality(assignment)
        f[1] = self.compactness(assignment)
        f[2] = self.minority_representation(assignment)
        f[3] = self.political_fairness(assignment)
        f[4] = self.competitiveness(assignment)
        f[5] = self.coi_preservation(assignment)
        f[6] = self.socioeconomic_parity(assignment)
        f[7] = self.environmental_justice(assignment)
        f[8] = self.trend_consideration(assignment)

        # Constraints
        g[0] = self.contiguity_constraint(assignment)

        return f, g

    def population_equality(self, assignment):
        """
        Calculates the population deviation across districts.
        """
        district_populations = np.array([
            self.population[assignment == i].sum() for i in range(self.num_districts)
        ])
        deviations = np.abs(district_populations - self.ideal_population) / self.ideal_population
        return deviations.max() * self.weights['population_equality']

    # Other methods remain largely the same, only adapting for the dynamic district count

def optimize_districting(data, seeds=[1, 2, 3, 4, 5]):
    """
    Runs the NSGA-III optimization algorithm multiple times with different seeds.
    """
    logger.info("Starting optimization with multi-start strategy.")

    all_assignments = []
    all_objectives = []

    for seed in seeds:
        logger.info(f"Optimization run with seed {seed}.")
        problem = RedistrictingProblem(data)
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
        mutation = PolynomialMutation(prob=Config.STOCHASTIC_MUTATION_PROB, eta=20)
        crossover = SBX(prob=0.9, eta=15)
        sampling = IntegerRandomSampling()

        algorithm = NSGA3(
            pop_size=Config.NSGA3_POPULATION_SIZE,
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )

        res = minimize(
            problem,
            algorithm,
            ('n_gen', Config.NSGA3_GENERATIONS),
            seed=seed,
            verbose=True
        )

        all_assignments.extend(res.X)
        all_objectives.extend(res.F)

    logger.info("Multi-start optimization completed.")
    return all_assignments, all_objectives

def generate_ensemble_plans(data, num_plans=10):
    """
    Generates an ensemble of districting plans.
    """
    logger.info("Generating ensemble of districting plans.")
    seeds = np.random.randint(1, 10000, size=num_plans)
    assignments, objectives = optimize_districting(data, seeds=seeds)
    return assignments
