import numpy as np

from true_graph.true_graph import TrueGraph
from simulation.simulation_graph import SimulationGraph

from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

"""
This module contains different metric function and can be extended to new ones.
All method signatures should look like this:
    def name(tG: TrueGraph, sG: SimulationGraph, params: dict) -> float:
Each sampling function should return a list of sampled edges
"""


def adjusted_randIndex(tG: TrueGraph, sG: SimulationGraph, params: dict) -> float:
    """
    Calculates the adjusted RandIndex of two clustered graphs

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: adjusted randIndex value
    """
    return metrics.adjusted_rand_score(tG.labels, sG.get_label_list(len(tG.labels)))


def purity(tG: TrueGraph, sG: SimulationGraph, params: dict) -> float:
    """
    Calculates the purity of two clustered graphs

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: purity value
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(
        tG.labels, sG.get_label_list(len(tG.labels)))
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def accuracy(tG: TrueGraph, sG: SimulationGraph, params: dict) -> float:
    """
    Calculates the accuracy with optimal mapping of two clustered graphs

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: accuracy value
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(
        tG.labels, sG.get_label_list(len(tG.labels)))

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def jensen_shannon_distance(tG: TrueGraph, sG: SimulationGraph, params: dict) -> float:
    """
    Calculates the Jensen Shannon Distance between two clustered graphs

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: Jensen Shannon Distance value
    """
    tG_nx_graph = tG.get_nx_graph_rep()
    # get comunnity probability vec
    tG_cluster_prob = [com[1] / tG_nx_graph.graph['number_nodes'] for com in tG_nx_graph.graph['community_sizes']]
    sG_cluster_prob = [com[1] / sG.G.graph['number_nodes'] for com in sG.G.graph['community_sizes']]

    # size them to same size
    for _ in range(len(sG_cluster_prob), len(tG_cluster_prob)):
        sG_cluster_prob.append(0.0)
    for _ in range(len(tG_cluster_prob), len(sG_cluster_prob)):
        tG_cluster_prob.append(0.0)

    return jensenshannon(tG_cluster_prob, sG_cluster_prob, base=2)

    pass
