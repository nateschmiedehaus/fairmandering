# tests/test_fairness_evaluation.py

import pytest
import pandas as pd
import numpy as np
from fairmandering.fairness_evaluation import evaluate_fairness

def test_evaluate_fairness():
    # Create mock data
    data = pd.DataFrame({
        'district': [0, 0, 1, 1],
        'P001001': [100, 150, 120, 130],
        'P005004': [30, 40, 35, 45],
        'votes_party_a': [60, 80, 70, 90],
        'votes_party_b': [40, 70, 50, 60],
        'geometry': [None, None, None, None],
        'median_income': [50000, 55000, 60000, 65000],
        'population_trend_mean': [125, 125, 125, 125],
        'population_trend_std': [0, 0, 0, 0]
    })

    assignment = np.array([0, 0, 1, 1])
    fairness = evaluate_fairness(data, assignment)

    assert 'population_equality' in fairness
    assert 'minority_representation' in fairness
    assert 'political_fairness' in fairness
    assert 'competitiveness' in fairness
    assert 'compactness' in fairness
    assert 'socioeconomic_parity' in fairness

    # Check values (based on mock data)
    assert fairness['population_equality'] == 0  # Equal populations
    assert fairness['minority_representation'] == 0.5  # (70 / 70)
    assert fairness['political_fairness'] == 0.25  # Average deviation from 0.5
    assert fairness['competitiveness'] == 0.25  # Average competitiveness
    assert fairness['compactness'] > 0  # Polsby-Popper score
    assert fairness['socioeconomic_parity'] == 0.1  # Example value
