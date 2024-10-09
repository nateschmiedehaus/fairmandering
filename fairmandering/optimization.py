# fairmandering/optimization.py

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions, get_mutation, get_sampling, get_crossover
from pymoo.optimize import minimize
import logging
from .config import Config
from sklearn.neighbors import kneighbors_graph
from joblib import Parallel, delayed
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.dominator import Dominator
from pymoo.util.misc import random_permuations

logger = logging.getLogger(__name__)

class RedistrictingProblem(Problem):
    """
    Defines the multi-objective optimization problem for fair redistricting.
    """

    def __init__(self, data, num_districts):
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

    def compactness(self, assignment):
        """
        Calculates the compactness of districts using Polsby-Popper measure.
        """
        compactness_scores = []
        for i in range(self.num_districts):
            district_geom = self.data[assignment == i].geometry.unary_union
            area = district_geom.area
            perimeter = district_geom.length
            if perimeter == 0:
                compactness_scores.append(0)
            else:
                compactness_scores.append(4 * np.pi * area / (perimeter ** 2))
        return -np.mean(compactness_scores) * self.weights['compactness']  # Negative because we minimize

    def minority_representation(self, assignment):
        """
        Calculates the representation of minority populations.
        """
        minority_populations = np.array([
            self.data.loc[assignment == i, 'P005004'].sum() for i in range(self.num_districts)
        ])
        total_minority_population = self.data['P005004'].sum()
        representation = minority_populations / total_minority_population
        return -representation.mean() * self.weights['minority_representation']

    def political_fairness(self, assignment):
        """
        Measures the political fairness based on past voting data.
        """
        party_a_votes = np.array([
            self.data.loc[assignment == i, 'votes_party_a'].sum() for i in range(self.num_districts)
        ])
        party_b_votes = np.array([
            self.data.loc[assignment == i, 'votes_party_b'].sum() for i in range(self.num_districts)
        ])
        total_votes = party_a_votes + party_b_votes
        vote_shares = party_a_votes / total_votes
        deviation = np.abs(vote_shares - 0.5)
        return deviation.mean() * self.weights['political_fairness']

    def competitiveness(self, assignment):
        """
        Measures the competitiveness of districts.
        """
        margins = np.array([
            np.abs(self.data.loc[assignment == i, 'votes_party_a'].sum() -
                   self.data.loc[assignment == i, 'votes_party_b'].sum())
            for i in range(self.num_districts)
        ])
        return margins.mean() * self.weights['competitiveness']

    def coi_preservation(self, assignment):
        """
        Measures how well communities of interest are preserved.
        """
        coi_counts = np.array([
            self.data.loc[assignment == i, 'coi'].nunique() for i in range(self.num_districts)
        ])
        return coi_counts.mean() * self.weights['coi_preservation']

    def socioeconomic_parity(self, assignment):
        """
        Measures socioeconomic parity across districts.
        """
        median_incomes = np.array([
            self.data.loc[assignment == i, 'median_rent'].mean() for i in range(self.num_districts)
        ])
        deviation = np.std(median_incomes) / np.mean(median_incomes)
        return deviation * self.weights['socioeconomic_parity']

    def environmental_justice(self, assignment):
        """
        Measures environmental hazard exposure across districts.
        """
        hazard_indices = np.array([
            self.data.loc[assignment == i, 'environmental_hazard_index'].mean() for i in range(self.num_districts)
        ])
        deviation = np.std(hazard_indices) / np.mean(hazard_indices)
        return deviation * self.weights['environmental_justice']

    def trend_consideration(self, assignment):
        """
        Considers demographic and political trends.
        """
        population_trends = np.array([
            self.data.loc[assignment == i, 'population_trend'].mean() for i in range(self.num_districts)
        ])
        deviation = np.std(population_trends) / np.mean(population_trends)
        return deviation * self.weights['trend_consideration']

    def contiguity_constraint(self, assignment):
        """
        Ensures that districts are contiguous.
        Returns 0 if constraint is satisfied, positive value otherwise.
        """
        violation = 0
        for i in range(self.num_districts):
            subgraph = self.adjacency_matrix[assignment == i][:, assignment == i]
            num_components = self.count_connected_components(subgraph)
            if num_components > 1:
                violation += num_components - 1
        return violation

    def count_connected_components(self, adjacency_submatrix):
        """
        Counts connected components in the adjacency submatrix.
        """
        n = adjacency_submatrix.shape[0]
        visited = np.zeros(n, dtype=bool)
        num_components = 0

        def dfs(v):
            visited[v] = True
            neighbors = adjacency_submatrix[v].nonzero()[1]
            for u in neighbors:
                if not visited[u]:
                    dfs(u)

        for v in range(n):
            if not visited[v]:
                dfs(v)
                num_components += 1

        return num_components

    def adjust_weights(self, generation):
        """
        Adjusts the weights of objectives adaptively based on the current generation.
        """
        max_generations = Config.NSGA3_GENERATIONS
        progress = generation / max_generations
        self.weights['population_equality'] = 1 + progress
        self.weights['compactness'] = 1 - progress
        # Adjust other weights as needed

def optimize_districting(data, seeds=[1, 2, 3, 4, 5]):
    """
    Runs the NSGA-III optimization algorithm multiple times with different seeds.
    """
    logger.info("Starting optimization with multi-start strategy.")

    all_assignments = []
    all_objectives = []

    for seed in seeds:
        logger.info(f"Optimization run with seed {seed}.")
        problem = RedistrictingProblem(data, Config.NUM_DISTRICTS)
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
        mutation = get_mutation("int_pm", prob=Config.STOCHASTIC_MUTATION_PROB)
        crossover = get_crossover("int_sbx", prob=0.9)
        sampling = get_sampling("int_random")

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
