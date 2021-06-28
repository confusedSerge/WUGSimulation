import numpy as np
import random

from collections import Counter
from graphs.base_graph import BaseGraph
from analysis.utils.metrics_utils import clean_labels
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

"""
This module contains different metric function and can be extended to new ones.
All method signatures should look like this:
    def name(tG: BaseGraph, sG: BaseGraph, params: dict) -> float:
Each sampling function should return a list of sampled edges
"""


def adjusted_randIndex(trueGraph: BaseGraph, simulatedGraph: BaseGraph, params: dict) -> float:
    """
    Calculates the adjusted RandIndex of two clustered graphs, 
        where only nodes of the simulated graph are considered.

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: adjusted randIndex value
    """
    return metrics.adjusted_rand_score(*clean_labels(trueGraph.labels, simulatedGraph.labels))


def purity(trueGraph: BaseGraph, simulatedGraph: BaseGraph, params: dict) -> float:
    """
    Calculates the purity of two clustered graphs, 
        where only nodes of the simulated graph are considered.

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: purity value
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(*clean_labels(trueGraph.labels, simulatedGraph.labels))
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def accuracy(trueGraph: BaseGraph, simulatedGraph: BaseGraph, params: dict) -> float:
    """
    Calculates the accuracy with optimal mapping of two clustered graphs,
        where only nodes of the simulated graph are considered.

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: accuracy value
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(*clean_labels(trueGraph.labels, simulatedGraph.labels))

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def inverse_jensen_shannon_distance(trueGraph: BaseGraph, simulatedGraph: BaseGraph, params: dict) -> float:
    """
    Calculates the Jensen Shannon Distance between two clustered graphs

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: Jensen Shannon Distance value
    """
    # get comunnity probability vec
    tG_cluster_prob = [com / trueGraph.get_number_nodes() for com in trueGraph.get_community_sizes()]
    sG_cluster_prob = [com / simulatedGraph.get_number_nodes() for com in simulatedGraph.get_community_sizes()]

    # size them to same size
    for _ in range(len(sG_cluster_prob), len(tG_cluster_prob)):
        sG_cluster_prob.append(0.0)
    for _ in range(len(tG_cluster_prob), len(sG_cluster_prob)):
        tG_cluster_prob.append(0.0)

    return 1 - jensenshannon(tG_cluster_prob, sG_cluster_prob, base=2)

@DeprecationWarning
def cluster_size_diff(true_graph: BaseGraph, simulation_graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse normalized sum of the difference between each graphs cluster sizes.
    
    Should not be used, as results, especially for #cluster == 1, do not make sense.
    Please use :cluster_diff_stripped():, as it strips down the ref clusters to the same number of nodes

    Args:
        :true_graph: first graph, which should model the reference graph
        :simulation_graph: second graph
        :returns float: value of the metric
    """
    tg_cluster_sizes = sorted(true_graph.get_community_sizes(), reverse=True)
    sg_clusters_sizes = sorted(simulation_graph.get_community_sizes(), reverse=True)

    # check lengths
    abs_diff = abs(len(tg_cluster_sizes) - len(sg_clusters_sizes))
    if len(tg_cluster_sizes) < len(sg_clusters_sizes):
        tg_cluster_sizes.extend([0] * abs_diff)
    else:
        sg_clusters_sizes.extend([0] * abs_diff)

    num_clusters = len(tg_cluster_sizes)
    c_sum = 0
    for i in range(num_clusters):
        c_sum += abs(sg_clusters_sizes[i] - tg_cluster_sizes[i])

    norm_factor = true_graph.get_number_nodes() + ((num_clusters - 2)/ num_clusters) * simulation_graph.get_number_nodes()
    norm_factor = norm_factor if norm_factor != 0 else 1
    
    return 1 - c_sum / norm_factor


def cluster_size_diff_stripped(true_graph: BaseGraph, simulation_graph: BaseGraph, params: dict) -> float:
    """
    Same as the cluster_diff function, but nodes not contained in the :simulation_graph: are removed/stripped
    from the reference for this calculation.

    Args:
        :true_graph: first graph, which should model the reference graph
        :simulation_graph: second graph
        :returns float: value of the metric
    """
    tg_cluster_sizes = []
    sg_clusters_sizes = sorted(simulation_graph.get_community_sizes(), reverse=True)

    not_added_nodes = set(true_graph.G.nodes()) - set(simulation_graph.G.nodes())

    for k, v in true_graph.get_community_nodes().items():
        tg_cluster_sizes.append(len(v) - len(set(v).intersection(not_added_nodes)))

    # check lengths
    abs_diff = abs(len(tg_cluster_sizes) - len(sg_clusters_sizes))
    if len(tg_cluster_sizes) < len(sg_clusters_sizes):
        tg_cluster_sizes.extend([0] * abs_diff)
    else:
        sg_clusters_sizes.extend([0] * abs_diff)

    num_clusters = len(tg_cluster_sizes)
    c_sum = 0
    for i in range(num_clusters):
        c_sum += abs(sg_clusters_sizes[i] - tg_cluster_sizes[i])

    norm_factor = simulation_graph.get_number_nodes() * (1 + (num_clusters - 2)/ num_clusters)
    norm_factor = norm_factor if norm_factor != 0 else 1
    
    return 1 - c_sum / norm_factor


def invers_entropy_distance(true_graph: BaseGraph, simulation_graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse entropy distance between a reference graph, where the clustering is known, and an unclustered graph.

    Args:
        :true_graph: first graph, which should model the reference graph
        :simulation_graph: second graph
        :threshold: which edge value should be regarded
        :returns float: value of the metric
    """
    threshold = params.get('threshold', 2.5)
    return 1 - abs(entropy_clustered(true_graph) / np.log2(true_graph.get_number_communities()) - entropy_unclustered(simulation_graph, threshold) / np.log2(simulation_graph.get_number_nodes()))

def invers_entropy_distance_clustered(true_graph: BaseGraph, simulation_graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse entropy distance between two graphs, where the clustering is known.

    Args:
        :true_graph: first graph, which should model the reference graph
        :simulation_graph: second graph
        :returns float: value of the metric
    """
    return 1 - abs(entropy_clustered(true_graph) / np.log2(true_graph.get_number_communities()) - entropy_clustered(simulation_graph) / np.log2(simulation_graph.get_number_communities()))


def entropy_unclustered(graph: BaseGraph, threshold: float) -> float:
    """
    Calculates the entropy of an unclustered graph.
    Args:
        :graph: on which to performe the evaluation
        :returns float: value of the metric
    """
    num_nodes = graph.get_number_nodes()
    node_num_edges_over_threshold = Counter([node for k, v in graph.get_weight_edge().items() if k >= threshold for t in v for node in t])

    s_sum = 0
    for i in graph.G.nodes():
        s_sum += np.log2((1 + node_num_edges_over_threshold.get(i, 0)) / num_nodes)

    return -(s_sum / num_nodes) 

def entropy_clustered(graph: BaseGraph) -> float:
    """
    Calculates the entropy of an clustered graph.
    Args:
        :param graph: on which to performe the evaluation
        :returns float: value of the metric
    """
    return entropy(graph.get_community_sizes(), base=2)

def cluster_num_diff(true_graph: BaseGraph, simulation_graph: BaseGraph, params: dict) -> float:
    """
    Returns the difference of numbers of clusters between two graphs (both have to be clustered).

    Args:
        :param true_graph: first clustered graph
        :param simulation_graph: second clustered graph
        :return float: diff
    """
    return abs(true_graph.get_number_communities() - simulation_graph.get_number_communities())

def apd(graph: BaseGraph, sample_size: int) -> float:
    """
    Calculates the APD (Average Pointwise Distance) of a graph (edge weights describing the distance).

    Args:
        :param graph: graph on which to calculate
        :param sample_size: the sample size to take from the graph
        :return float: apd of the sample
    """
    sampled_edge_list = []

    for _ in range(sample_size):
        u, v = sorted(random.sample(graph.G.nodes(), 2))
        sampled_edge_list.append(graph.get_edge(u, v))

    return sum(sampled_edge_list) / sample_size
    
def hpd(graph: BaseGraph, sample_size: int) -> float:
    """
    Calculates the entropy of the sampled pointwise distribution of edge weights.

    Args:
        :param graph: graph on which to calculate
        :param sample_size: the sample size to take from the graph
        :return float: hpd of the sample
    """
    sampled_edge_list = []

    for _ in range(sample_size):
        u, v = sorted(random.sample(graph.G.nodes(), 2))
        sampled_edge_list.append(graph.get_edge(u, v))

    count_edges = [v for k, v in Counter(sampled_edge_list).items()]

    return entropy(count_edges, base=2)

def hpd_normalized(graph: BaseGraph, sample_size: int) -> float:
    """
    Calculates the normalized entropy of the sampled pointwise distribution of edge weights.

    Args:
        :param graph: graph on which to calculate
        :param sample_size: the sample size to take from the graph
        :return float: hpd of the sample
    """
    sampled_edge_list = []

    for _ in range(sample_size):
        u, v = sorted(random.sample(graph.G.nodes(), 2))
        sampled_edge_list.append(graph.get_edge(u, v))

    count_edges = [v for k, v in Counter(sampled_edge_list).items()]

    return entropy(count_edges, base=2) / len(count_edges)
