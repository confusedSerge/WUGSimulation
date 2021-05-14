import numpy as np

from graphs.base_graph import BaseGraph
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

"""
This module contains different metric function and can be extended to new ones.
All method signatures should look like this:
    def name(tG: BaseGraph, sG: BaseGraph, params: dict) -> float:
Each sampling function should return a list of sampled edges
"""


def adjusted_randIndex(trueGraph: BaseGraph, simulatedGraph: BaseGraph, params: dict) -> float:
    """
    Calculates the adjusted RandIndex of two clustered graphs

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: adjusted randIndex value
    """
    return metrics.adjusted_rand_score(trueGraph.labels, simulatedGraph.labels)


def purity(trueGraph: BaseGraph, simulatedGraph: BaseGraph, params: dict) -> float:
    """
    Calculates the purity of two clustered graphs

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: purity value
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(trueGraph.labels, simulatedGraph.labels)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def accuracy(trueGraph: BaseGraph, simulatedGraph: BaseGraph, params: dict) -> float:
    """
    Calculates the accuracy with optimal mapping of two clustered graphs

    Args:
        :param tG: first graph
        :param sG: second graph
        :returns float: accuracy value
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(trueGraph.labels, simulatedGraph.labels)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def jensen_shannon_distance(trueGraph: BaseGraph, simulatedGraph: BaseGraph, params: dict) -> float:
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

    return jensenshannon(tG_cluster_prob, sG_cluster_prob, base=2)
