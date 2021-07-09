from graphs.base_graph import BaseGraph
from simulation.stopping.utils.stopping_utils import apd as _apd
from simulation.stopping.utils.stopping_utils import entropy_approximation as _ea

"""
This module provides a more advance use of stopping criterions, like using time step information.
If any information is being used, a reset function has to be provided
"""

# ===Convergence Criteria===

_apd_time_steps = []

def apd_convergence(graph: BaseGraph, params: dict) -> bool:
    """
    Calculates the APD (Average Pointwise Distance) of a graph (edge weights describing the distance)
    and checks if the absolute change between a given timeframe of APD is not greater than a given treshold. 
    # TODO: what should the x value be? timesteps, judgements, ...current is timesteps

    Args:
        :param graph: graph on which to calculate
        :param sample_size: the sample size to take from the graph
        :param timesteps: timesteps to take into account
        :param threshold: under which threshold the function should resolve to true
        :return bool: if change is under the threshold
    """
    # ===Guard Phase===
    sample_size = params.get('sample_size', 50)
    assert type(sample_size) == int

    timesteps = params.get('timesteps', 5)
    assert type(timesteps) == int

    threshold = params.get('threshold', .1)
    assert type(threshold) == float
    
    global _apd_time_steps
    # ===End Guard Phase===

    _apd_time_steps.append((len(_apd_time_steps), _apd(graph, sample_size)))

    if len(_apd_time_steps) < timesteps: return False

    w_slope = 0
    for i in range(1, timesteps):
        w_slope += abs((_apd_time_steps[-1][1] - _apd_time_steps[-1 - i][1]) / (_apd_time_steps[-1][0] - _apd_time_steps[-1 - i][0])) / i

    return w_slope < threshold

def apd_convergence_reset():
    global _apd_time_steps
    _apd_time_steps.clear()

_entropy_time_steps = []

def entropy_approx_convergence(graph: BaseGraph, params: dict) -> bool:
    """
    Calculates the approximate entropy of an unclustered graph
    and checks if the absolute change between a given timeframe of Entropy is not greater than a given treshold.
    Args:
        :param graph: on which to performe the evaluation
        :param threshold_entropy: which edges to consider
        :param timesteps: timesteps to take into account
        :param threshold_conv: under which threshold the function should resolve to true
        :returns float: if change is under the threshold
    """
    # ===Guard Phase===
    threshold_entropy = params.get('threshold_entropy', 2.5)
    assert type(threshold_entropy) == float

    timesteps = params.get('timesteps', 5)
    assert type(timesteps) == int

    threshold_conv = params.get('threshold_conv', .1)
    assert type(threshold_conv) == float

    global _entropy_time_steps
    # ===End Guard Phase===

    _entropy_time_steps.append((len(_entropy_time_steps), _ea(graph, threshold_entropy)))

    if len(_entropy_time_steps) < timesteps: return False

    w_slope = 0
    for i in range(1, timesteps):
        w_slope += abs((_entropy_time_steps[-1][1] - _entropy_time_steps[-1 - i][1]) / (_entropy_time_steps[-1][0] - _entropy_time_steps[-1 - i][0])) / i

    return w_slope < threshold_conv

def entropy_approx_convergence_reset():
    global _entropy_time_steps
    _entropy_time_steps.clear()
