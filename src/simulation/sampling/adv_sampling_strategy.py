from graphs.base_graph import BaseGraph

from simulation.sampling.utils.dwug_sampling import dwug_sampling as u_dwug_sampling

"""
This module contains different advanced sampling functions and can be extended to new ones.
All method signatures should look like this:
    def name(graph: BaseGraph, annotated_graph: Base, params: dict) -> list:
Each sampling function should return a list of sampled edges
"""


def dwug_sampling(graph: BaseGraph, annotated_graph: BaseGraph, params: dict) -> list:
    """
    Uses the DWUG sampling strategy, as described in the paper.

    Args:
        :param graph: graph on which to sample edge weight
        :param annotated_graph: simulation graph
        :param percentage_nodes: percentage of nodes to add this round
        :param percentage_edges: percentage of edges to add this round
        :param min_size_mc: minimum size of cluster to be considered as multi-cluster
        :param num_flag: if :percentage_nodes: & :percentage_edges: are the actual number of nodes/edges to be used  (optional)
        :return sampled_edge_list: sampled edges with weights as [(u, v, w)...]
    """
    # ===Guard Phase===
    assert isinstance(graph, BaseGraph)
    assert isinstance(annotated_graph, BaseGraph)

    percentage_nodes = params.get('percentage_nodes', None)
    assert (type(percentage_nodes) == float and 0.0 <= percentage_nodes <= 1.0) or (
        type(percentage_nodes) == int and 0 <= percentage_nodes <= graph.get_number_nodes())

    percentage_edges = params.get('percentage_edges', None)
    assert (type(percentage_edges) == float and 0.0 <= percentage_edges <= 1.0) or (
        type(percentage_edges) == int and 0 <= percentage_edges <= graph.get_number_edges())
    min_size_mc = params.get('min_size_mc', None)
    assert type(min_size_mc) == int

    num_flag = params.get('num_flag', False)
    assert type(num_flag) == bool
    # ===Guard Phase over===

    return u_dwug_sampling(graph, annotated_graph, percentage_nodes, percentage_edges, min_size_mc, num_flag)
