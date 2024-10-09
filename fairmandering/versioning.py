# fairmandering/versioning.py

import json
import os
import datetime
import logging

logger = logging.getLogger(__name__)

def save_plan(assignment, metadata, version):
    """
    Saves a districting plan to a file with metadata.

    Args:
        assignment (array): The district assignment.
        metadata (dict): Additional information about the plan.
        version (str): The version identifier.

    Returns:
        str: The file path of the saved plan.
    """
    plan = {
        'version': version,
        'timestamp': datetime.datetime.now().isoformat(),
        'metadata': metadata,
        'assignment': assignment.tolist()
    }
    filename = f"plans/plan_{version}.json"
    os.makedirs('plans', exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(plan, f)
    logger.info(f"Plan saved as {filename}.")
    return filename

def load_plan(filename):
    """
    Loads a districting plan from a file.

    Args:
        filename (str): The file path of the plan.

    Returns:
        dict: The plan data.
    """
    with open(filename, 'r') as f:
        plan = json.load(f)
    logger.info(f"Plan loaded from {filename}.")
    return plan

def compare_plans(plan1, plan2):
    """
    Compares two districting plans and highlights differences.

    Args:
        plan1 (dict): The first plan.
        plan2 (dict): The second plan.

    Returns:
        dict: A summary of differences.
    """
    differences = {
        'version1': plan1['version'],
        'version2': plan2['version'],
        'differences': []
    }
    # Compare assignments
    assignment1 = plan1['assignment']
    assignment2 = plan2['assignment']
    if len(assignment1) != len(assignment2):
        differences['differences'].append('Assignments have different lengths.')
    else:
        diff_count = sum(1 for a, b in zip(assignment1, assignment2) if a != b)
        differences['differences'].append(f"{diff_count} units assigned to different districts.")

    # Compare metadata
    metadata_differences = {k: (plan1['metadata'].get(k), plan2['metadata'].get(k))
                            for k in set(plan1['metadata']) | set(plan2['metadata'])
                            if plan1['metadata'].get(k) != plan2['metadata'].get(k)}
    if metadata_differences:
        differences['differences'].append(f"Metadata differences: {metadata_differences}")

    logger.info(f"Compared plans {plan1['version']} and {plan2['version']}.")
    return differences
