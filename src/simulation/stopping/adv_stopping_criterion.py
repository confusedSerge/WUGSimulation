from graphs.base_graph import BaseGraph
from simulation.stopping.utils.stopping_utils import apd as _apd
from simulation.stopping.utils.stopping_utils import entropy_approximation as _ea
from simulation.stopping.utils.stopping_utils import rmse_mean as _rmse
from simulation.stopping.utils.stopping_utils import clean_labels as _cl
from simulation.stopping.utils.stopping_utils import length_padding as _lp

from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import jensenshannon as _js_distance

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

    _rmse_value = _rmse(_apd_time_steps[-timesteps:])

    return _rmse_value < threshold

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

    _rmse_value = _rmse(_entropy_time_steps[-timesteps:])

    return _rmse_value < threshold_conv

def entropy_approx_convergence_reset():
    global _entropy_time_steps
    _entropy_time_steps.clear()

_num_clusters = []

def cluster_size_change_conv(graph: BaseGraph, params: dict) -> bool:
    """

    """
    # ===End Guard Phase===
    timesteps = params.get('timesteps', 5)
    assert type(timesteps) == int

    threshold = params.get('threshold', .1)
    assert type(threshold) == float
    
    global _num_clusters
    # ===End Guard Phase===
    _num_clusters.append(graph.get_number_communities)
    
    if len(_num_clusters) < timesteps: return False

    _rmse_value = _rmse(_num_clusters[-timesteps:])

    return _rmse_value < threshold

def cluster_size_change_conv_reset():
    global _num_clusters
    _num_clusters.clear()

_rand_index = []
_last_label = None

def _rand_index_conv(graph: BaseGraph, params: dict) -> bool:
    """

    """
    # ===End Guard Phase===
    timesteps = params.get('timesteps', 5)
    assert type(timesteps) == int

    threshold = params.get('threshold', .1)
    assert type(threshold) == float
    
    global _rand_index
    global _last_label
    # ===End Guard Phase===
    _rand_index.append(adjusted_rand_score(*_cl(_last_label, graph.labels)))
    _last_label = graph.labels.copy()

    if len(_rand_index) < timesteps: return False

    _rmse_value = _rmse(_rand_index[-timesteps:])

    return _rmse_value < threshold

def _rand_index_conv_reset():
    global _rand_index
    global _last_label

    _rand_index.clear()
    _last_label = None


_jsd = []
_last_cluster = None

def _jsd_conv(graph: BaseGraph, params: dict) -> bool:
    """

    """
    # ===End Guard Phase===
    timesteps = params.get('timesteps', 5)
    assert type(timesteps) == int

    threshold = params.get('threshold', .1)
    assert type(threshold) == float
    
    global _jsd
    global _last_cluster
    # ===End Guard Phase===
    cl1, cl2 = _lp(_last_cluster, graph.get_community_sizes())
    _jsd.append(_js_distance(cl1, cl2, base=2))
    _last_cluster = graph.get_community_sizes()

    if len(_jsd) < timesteps: return False

    _rmse_value = _rmse(_jsd[-timesteps:])

    return _rmse_value < threshold

def _jsd_conv_reset():
    global _jsd
    global _last_cluster

    _jsd.clear()
    _last_cluster = None

