import numpy as np
import random

from collections import Counter
from graphs.base_graph import BaseGraph
from networkx import Graph
from copy import deepcopy

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
        # if node_num_edges_over_threshold.get(i, 0) != 0:

    return -(s_sum / num_nodes)


def build_full_m(reference_graph: BaseGraph, graph: BaseGraph) -> BaseGraph:
    # TODO: Massive reference problem in this function. This is a work around
    reference_graph = deepcopy(reference_graph)
    graph = deepcopy(graph)

    new_m = BaseGraph()
    new_m.G = graph.G.copy()
    new_m.G.graph['weight_edge'] = dict(graph.G.graph['weight_edge'])

    def P2M(x): return x + len(graph.G.nodes())
    def M2P(x): return x - len(graph.G.nodes())

    for v in reference_graph.G.nodes():
        _cn = dict(new_m.G.nodes())
        for u in _cn:
            if u == v:
                new_m.add_edge(u, P2M(v), weight=4)
            else:
                _u = u if u < len(graph.G.nodes()) else M2P(u)
                w = reference_graph.get_edge(_u, v)
                new_m.add_edge(u, P2M(v), weight=w)

    return new_m


def build_m(reference_graph: BaseGraph, graph: BaseGraph) -> BaseGraph:
    # TODO: Massive reference problem in this function. This is a work around
    reference_graph = deepcopy(reference_graph)
    graph = deepcopy(graph)

    new_m = BaseGraph()
    new_m.G = graph.G.copy()
    new_m.G.graph['weight_edge'] = dict(graph.G.graph['weight_edge'])

    def P2M(x): return x + len(graph.G.nodes())
    def M2P(x): return x - len(graph.G.nodes())

    for i, v in enumerate(reference_graph.G.nodes()):
        if i >= len(graph.G.nodes()):
            break
        _cn = dict(new_m.G.nodes())
        for u in _cn:
            if u == v:
                new_m.add_edge(u, P2M(v), weight=4)
            else:
                _u = u if u < len(graph.G.nodes()) else M2P(u)
                w = reference_graph.get_edge(_u, v)
                new_m.add_edge(u, P2M(v), weight=w)

    return new_m


def build_m_modified_ref(reference_graph: BaseGraph, graph: BaseGraph) -> (BaseGraph, BaseGraph):
    # TODO: Massive reference problem in this function. This is a work around
    reference_graph = deepcopy(reference_graph)
    graph = deepcopy(graph)

    new_m = BaseGraph()
    new_m.G = graph.G.copy()
    new_m.G.graph['weight_edge'] = dict(graph.G.graph['weight_edge'])

    new_mrf = BaseGraph()

    def P2M(x): return x + len(graph.G.nodes())
    def M2P(x): return x - len(graph.G.nodes())

    _added_from_ref = []
    for i, v in enumerate(reference_graph.G.nodes()):
        if i >= len(graph.G.nodes()):
            break
        _cn = dict(new_m.G.nodes())
        for u in _cn:
            if u == v:
                new_m.add_edge(u, P2M(v), weight=4)
            else:
                _u = u if u < len(graph.G.nodes()) else M2P(u)
                w = reference_graph.get_edge(_u, v)
                new_m.add_edge(u, P2M(v), weight=w)
        _added_from_ref.append(v)

    if len(_added_from_ref):
        new_mrf.G.add_node(P2M(_added_from_ref[0]))
    for i in range(len(_added_from_ref)):
        for j in range(i + 1, len(_added_from_ref)):
            w = reference_graph.get_edge(_added_from_ref[i], _added_from_ref[j])
            new_mrf.add_edge(P2M(_added_from_ref[i]), P2M(_added_from_ref[j]), w)

    return new_m, new_mrf


def kld(p: BaseGraph, q: BaseGraph, threshold: float) -> float:
    num_nodes_p = p.get_number_nodes()
    num_nodes_q = q.get_number_nodes()
    node_num_edges_over_threshold_p = Counter([node for k, v in p.get_weight_edge().items() if k >= threshold for t in v for node in t])
    node_num_edges_over_threshold_q = Counter([node for k, v in q.get_weight_edge().items() if k >= threshold for t in v for node in t])

    s_sum = 0
    for i in p.G.nodes():
        p_si = (1 + node_num_edges_over_threshold_p.get(i, 0)) / num_nodes_p
        q_si = (1 + node_num_edges_over_threshold_q.get(i, 0)) / num_nodes_q
        s_sum += np.log2(p_si / q_si)

    return s_sum / num_nodes_p


def kld_normalized(p: BaseGraph, q: BaseGraph, threshold: float) -> float:
    return 1 - np.exp(-kld(p, q, threshold))


def random_sample_cluster(graph: BaseGraph, sample_size: int) -> list:
    """
    Random samples cluster distribution from provided graph.

    Args:
        :param graph: Clustered Graph
        :param sample_size: Number of nodes to sample with replacement
    """
    community_sizes: list = [0] * graph.get_number_communities()

    for node in np.random.choice(graph.G.nodes(), sample_size):
        community_sizes[graph.get_community_of_node(node)] += 1

    return community_sizes


def pertubate(graph: BaseGraph, range_judgements: tuple = (1, 4), share: float = 0.1) -> BaseGraph:
    new_graph: BaseGraph = BaseGraph()
    edges = list(graph.G.edges())

    for edge in edges:
        new_graph.add_edge(*edge, graph.get_edge(*edge))

    try:
        mod_edge = [edges[i] for i in np.random.choice(len(edges), int(len(edges) * share), replace=False)]
    except ValueError:
        mod_edge = edges

    for edge in mod_edge:
        we = new_graph.get_edge_weight_history(*edge)
        we.append(random.randint(range_judgements[0], range_judgements[1]))
        new_graph.add_edge(*edge, np.median(we))

    return new_graph


def gen_labels(graph: BaseGraph) -> list:
    labels = [-1] * (max(graph.G.nodes()) + 1)

    for k, v in graph.G.graph['node_community'].items():
        labels[k] = v
    return labels
