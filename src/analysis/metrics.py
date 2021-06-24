import numpy as np

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

def cluster_diff(true_graph: BaseGraph, simulation_graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse normalized sum of the difference between each graphs cluster sizes.

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

    c_sum = 0
    for i in range(len(tg_cluster_sizes)):
        c_sum += abs(sg_clusters_sizes[i] - tg_cluster_sizes[i])
    
    return 1 - c_sum / true_graph.get_number_nodes()


def cluster_diff_stripped(true_graph: BaseGraph, simulation_graph: BaseGraph, params: dict) -> float:
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

    for k, v in true_graph.get_community_nodes():
        tg_cluster_sizes.append(len(v) - len(set(v).intersection(not_added_nodes)))

    # check lengths
    abs_diff = abs(len(tg_cluster_sizes) - len(sg_clusters_sizes))
    if len(tg_cluster_sizes) < len(sg_clusters_sizes):
        tg_cluster_sizes.extend([0] * abs_diff)
    else:
        sg_clusters_sizes.extend([0] * abs_diff)

    c_sum = 0
    for i in range(len(tg_cluster_sizes)):
        c_sum += abs(sg_clusters_sizes[i] - tg_cluster_sizes[i])
    
    return 1 - c_sum / simulation_graph.get_number_nodes()


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
    return 1 - abs(entropy_clustered(true_graph) / np.log2(true_graph.get_number_nodes()) - entropy_unclustered(simulation_graph, threshold) / np.log2(simulation_graph.get_number_nodes()))

def invers_entropy_distance_clustered(true_graph: BaseGraph, simulation_graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse entropy distance between two graphs, where the clustering is known.

    Args:
        :true_graph: first graph, which should model the reference graph
        :simulation_graph: second graph
        :returns float: value of the metric
    """
    return 1 - abs(entropy_clustered(true_graph) / np.log2(true_graph.get_number_nodes()) - entropy_clustered(simulation_graph) / np.log2(simulation_graph.get_number_nodes()))


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
        :graph: on which to performe the evaluation
        :returns float: value of the metric
    """
    return entropy(graph.get_community_sizes(), base=2)

    