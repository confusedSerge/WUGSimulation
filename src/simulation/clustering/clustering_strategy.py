import numpy as np

from graphs.base_graph import BaseGraph
from simulation.clustering.utils.cluster_correlation_search import cluster_correlation_search as _old_cluster_correlation_search
from simulation.clustering.utils.new_cluster_correlation_search import cluster_correlation_search as _new_cluster_correlation_search
from chinese_whispers import chinese_whispers as _chinese_whispers
from chinese_whispers import aggregate_clusters as _aggregate_clusters
from community.community_louvain import best_partition as _louvain_partition
from networkx import connected_components
from simulation.clustering.utils.utils import sort_len_nodes as _clsort
from simulation.clustering.utils.utils import louvain_cluster_sort as _lvsort

"""
This module contains different clustering functions and can be extended to new ones.
All method signatures should look like this:
    def name(graph: BaseGraph, params: dict) -> dict:
Each clustering function should only return a dictionary with label-node key-value pairs
"""


@DeprecationWarning
def correlation_clustering(graph: BaseGraph, params: dict) -> dict:
    """
    This clustering implementation uses the clustering algorithm mentioned in:
        'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'

    Args:
        :param graph: graph to cluster
        :param weights: weights to be used for clustering
        :param s: maximal number of senses a word can have
        :param max_attempts: number of restarts for optimization
        :param max_iters: number of iterations for optimization
        :return labels: dict with label-node key-value pairs
    """
    # ===Guard Phase===
    s = params.get('s', 10)
    assert type(s) == int

    weights = params.get('weights', 'edge_weight')
    assert type(weights) == str

    max_attempts = params.get('max_attempts', 200)
    assert type(max_attempts) == int

    max_iters = params.get('max_iters', 500)
    assert type(max_iters) == int
    # ===Guard Phase===

    clusters = _old_cluster_correlation_search(G=graph.get_nx_graph_copy(
        weights), s=s, max_attempts=max_attempts, max_iters=max_iters)

    community_node = {}
    for cluster_id, cluster in enumerate(clusters):
        community_node[cluster_id] = list(cluster)

    return community_node


def new_correlation_clustering(graph: BaseGraph, params: dict) -> dict:
    """
    This clustering implementation uses the clustering algorithm mentioned in:
        'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'

    Args:
        :param graph: graph to clusters
        :param weights: weights to be used for clustering
        :param s: maximal number of senses a word can have
        :param max_attempts: number of restarts for optimization
        :param max_iters: number of iterations for optimization
        :param split_flag: optional flag, if non evidence cluster should be splitted
        :param ru_old_cluster: optional flag, if old cluster should be reused
        :return labels: dict with label-node key-value pairs
    """
    # ===Guard Phase===
    s = params.get('s', 10)
    assert type(s) == int

    weights = params.get('weights', 'edge_weight')
    assert type(weights) == str

    max_attempts = params.get('max_attempts', 200)
    assert type(max_attempts) == int

    max_iters = params.get('max_iters', 500)
    assert type(max_iters) == int or max_iters == np.inf

    split_flag = params.get('split_flag', True)
    assert type(split_flag) == bool

    ru_old_cluster = params.get('ru_old_cluster', False)
    assert type(ru_old_cluster) == bool
    # ===Guard Phase===

    initial = [set(v) for k, v in sorted(graph.get_community_nodes().items())] if ru_old_cluster else []

    clusters, stats = _new_cluster_correlation_search(G=graph.get_nx_graph_copy(
        weights), s=s, max_attempts=max_attempts, max_iters=max_iters, initial=initial, split_flag=split_flag)

    community_node = {}
    for cluster_id, cluster in enumerate(clusters):
        community_node[cluster_id] = list(cluster)

    return _clsort(community_node)


def connected_components_clustering(graph: BaseGraph, params: dict) -> dict:
    """
    Apply connected_component clustering.

    Args"
        :param graph: graph to clusters
        :param weights: weights to be used for clustering
        :return labels: dict with label-node key-value pairs
    """
    weights = params.get('weights', 'edge_weight')
    assert type(weights) == str

    G = graph.get_nx_graph_copy(weights)

    def is_non_value(x):
        return np.isnan(x)

    edges_negative = [(i, j) for (i, j) in G.edges() if G[i]
                      [j]['weight'] < 0.0 or is_non_value(G[i][j]['weight'])]
    G.remove_edges_from(edges_negative)
    return _clsort({i: list(component) for i, component in enumerate(connected_components(G))})


def chinese_whisper_clustering(graph: BaseGraph, params: dict) -> dict:
    """
    Perform 'chinese whispers' clustering.

    Args:
        :param graph: graph to clusters
        :param weights: weights to be used for clustering
        :return labels: dict with label-node key-value pairs
    """
    weights = params.get('weights', 'edge_weight')
    assert type(weights) == str

    G = graph.get_nx_graph_copy(weights)
    _cw = _aggregate_clusters(_chinese_whispers(G, weighting='top', iterations=20))
    return _clsort({k: list(v) for k, v in _cw.items()})


def louvain_method_clustering(graph: BaseGraph, params: dict) -> dict:
    """
    Perform 'louvain method' clustering.

    Args:
        :param graph: graph to clusters
        :param weights: weights to be used for clustering
        :return labels: dict with label-node key-value pairs
    """
    weights = params.get('weights', 'edge_weight')
    assert type(weights) == str

    G = graph.get_nx_graph_copy(weights)
    # partition: list = sorted(_louvain_partition(G).items(), key=lambda x: x[0])
    return _lvsort(_louvain_partition(G))
