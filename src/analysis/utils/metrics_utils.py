import numpy as np
from collections import Counter
from graphs.base_graph import BaseGraph

"""
Utility functions for metric script
"""

def clean_labels(ref_labels: list, current_labels: list) -> (list, list):
    """
    Uses the reference labels and current labels to create two list of the same size,
        only holding labels, where current labels is not -1.

    Args:
        :param ref_labels: containing real labels
        :param current_labels: containing simulated labels
        :return (list, list): new calculated lists, (ref, current)
    """
    assert len(ref_labels) == len(current_labels)

    new_ref = []
    new_crr = []

    for i in range(len(ref_labels)):
        if current_labels[i] != -1:
            new_ref.append(ref_labels[i])
            new_crr.append(current_labels[i])
    
    return new_ref, new_crr


def entropy_approximation(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the approximate entropy of an unclustered graph.
    Args:
        :param graph: on which to performe the evaluation
        :param threshold: which edges to consider
        :returns float: value of the metric
    """
    threshold = params.get('threshold', 2.5)

    num_nodes = graph.get_number_nodes()
    node_num_edges_over_threshold = Counter([node for k, v in graph.get_weight_edge().items() if k >= threshold for t in v for node in t])

    s_sum = 0
    for i in graph.G.nodes():
        s_sum += np.log2((1 + node_num_edges_over_threshold.get(i, 0)) / num_nodes)

    return -(s_sum / num_nodes) 

def entropy_approximation_combined(graph_one: BaseGraph, graph_two: BaseGraph, params: dict) -> float:
    """
    Calculates the approximate entropy of two unclustered graphs.
    Args:
        :param graph: on which to performe the evaluation
        :param threshold: which edges to consider
        :returns float: value of the metric
    """
    threshold = params.get('threshold', 2.5)

    node_num_edges_over_threshold = Counter([node for k, v in graph_one.get_weight_edge().items() if k >= threshold for t in v for node in t])
    node_num_edges_over_threshold += Counter([node for k, v in graph_two.get_weight_edge().items() if k >= threshold for t in v for node in t])
    for k, v in node_num_edges_over_threshold.items():
        node_num_edges_over_threshold[k] = v / 2
    
    num_nodes = (graph_one.get_number_nodes() + graph_two.get_number_nodes()) / 2

    s_sum = 0
    for i in set(graph_one.nodes()).union(set(graph_two.nodes())):
        s_sum += np.log2((1 + node_num_edges_over_threshold.get(i, 0)) / num_nodes)

    return -(s_sum / num_nodes) 