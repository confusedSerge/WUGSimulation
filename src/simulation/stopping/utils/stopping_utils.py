import numpy as np
import random
from collections import Counter
from graphs.base_graph import BaseGraph

def check_connectivity_two_clusters(edge_list: list, f_community: list, s_community: list, min_connections: int) -> bool:
    """
    Check if minimum connections between both communities.

    Args:
        :param edge_list: edge list on which to check
        :param f_community: first node list
        :param f_community: second node list
        :param min_connections: minimum connection so that it evaluates to True
    """
    count = 0
    for u in f_community:
        for v in s_community:
            if (u, v) in edge_list:
                count += 1
                if count >= min_connections:
                    return True

    return False

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

def entropy_approximation(graph: BaseGraph, threshold: float) -> float:
    """
    Calculates the approximate entropy of an unclustered graph.
    Args:
        :param graph: on which to performe the evaluation
        :param threshold: which edges to consider
        :returns float: value of the metric
    """
    num_nodes = graph.get_number_nodes()
    node_num_edges_over_threshold = Counter([node for k, v in graph.get_weight_edge().items() if k >= threshold for t in v for node in t])

    s_sum = 0
    for i in graph.G.nodes():
        s_sum += np.log2((1 + node_num_edges_over_threshold.get(i, 0)) / num_nodes)

    return -(s_sum / num_nodes) 