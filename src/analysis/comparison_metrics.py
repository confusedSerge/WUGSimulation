import numpy as np
import random

from collections import Counter
from graphs.base_graph import BaseGraph
from analysis.utils.metrics_utils import clean_labels
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from analysis.utils.metrics_utils import entropy_approximation
from analysis.utils.metrics_utils import entropy_approximation_combined

"""
This module contains different metric function and can be extended to new ones.
method signature expected:
    def name(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
"""


def adjusted_rand_index(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the adjusted RandIndex of two clustered graphs, 
        where only nodes of the second graph are considered.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: adjusted randIndex value
    """
    return metrics.adjusted_rand_score(*clean_labels(reference_graph.labels, graph.labels))

def purity(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the purity of two clustered graphs, 
        where only nodes of the second graph are considered.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: purity value
    """
    contingency_matrix = metrics.cluster.contingency_matrix(*clean_labels(reference_graph.labels, graph.labels))
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def accuracy(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the accuracy with optimal mapping of two clustered graphs,
        where only nodes of the second graph are considered.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: accuracy value
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(*clean_labels(reference_graph.labels, graph.labels))

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

def inverse_jensen_shannon_distance(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the Jensen Shannon Distance between two clustered graphs

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: Jensen Shannon Distance value
    """
    # get community probability vec
    ref_cluster_prob = [com / reference_graph.get_number_nodes() for com in reference_graph.get_community_sizes()]
    g_cluster_prob = [com / graph.get_number_nodes() for com in graph.get_community_sizes()]

    # size them to same size
    for _ in range(len(g_cluster_prob), len(ref_cluster_prob)):
        g_cluster_prob.append(0.0)
    for _ in range(len(ref_cluster_prob), len(g_cluster_prob)):
        ref_cluster_prob.append(0.0)

    return 1 - jensenshannon(ref_cluster_prob, g_cluster_prob, base=2)

def inverse_jensen_shannon_divergence(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the Jensen Shannon Divergence between two clustered graphs

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: Jensen Shannon Distance value
    """
    return 1 - (1 - inverse_jensen_shannon_distance(reference_graph, graph, params))**2

def invers_entropy_distance_clustered(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse entropy distance between two graphs, where the clustering is known.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: value of the metric
    """
    return 1 - abs(entropy(graph.get_community_sizes(), base=2) / np.log2(reference_graph.get_number_communities()) - entropy(graph.get_community_sizes(), base=2) / np.log2(graph.get_number_communities()))

def cluster_size_diff_stripped(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse normalized sum of the difference between each graphs cluster sizes.
    Nodes not contained in the :reference_graph: are removed/stripped from the reference for this calculation.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: value of the metric
    """
    ref_cluster_sizes = []
    g_clusters_sizes = sorted(graph.get_community_sizes(), reverse=True)

    not_added_nodes = set(reference_graph.G.nodes()) - set(graph.G.nodes())

    for k, v in reference_graph.get_community_nodes().items():
        ref_cluster_sizes.append(len(v) - len(set(v).intersection(not_added_nodes)))

    # check lengths
    abs_diff = abs(len(ref_cluster_sizes) - len(g_clusters_sizes))
    if len(ref_cluster_sizes) < len(g_clusters_sizes):
        ref_cluster_sizes.extend([0] * abs_diff)
    else:
        g_clusters_sizes.extend([0] * abs_diff)

    num_clusters = len(ref_cluster_sizes)
    c_sum = 0
    for i in range(num_clusters):
        c_sum += abs(g_clusters_sizes[i] - ref_cluster_sizes[i])

    norm_factor = graph.get_number_nodes() * (1 + (num_clusters - 2)/ num_clusters)
    norm_factor = norm_factor if norm_factor != 0 else 1
    
    return 1 - c_sum / norm_factor

def cluster_num_diff(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Returns the difference of numbers of clusters between two graphs (both have to be clustered).

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :return float: diff
    """
    return abs(reference_graph.get_number_communities() - graph.get_number_communities())


def jensen_shannon_distance_approximation(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the Jensen-Shanon-Distance based on entropy.
    To calculate the entropy of a graph, use either :func entropy_approximation: or :func entropy_clustered:

    Args:
        :param graph_one_entropy: entropy of the first graph
        :param graph_two_entropy: entropy of the second graph
        :param combined_graph_entropy: entropy of the combined graph
        :return float: jsd based on entropy
    """
    return entropy_approximation_combined(reference_graph, graph) - (entropy_approximation(reference_graph) + entropy_approximation(graph)) / 2
