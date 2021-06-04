from graphs.base_graph import BaseGraph
from simulation.clustering.utils.cluster_correlation_search import cluster_correlation_search as old_cluster_correlation_search
from simulation.clustering.utils.new_cluster_correlation_search import cluster_correlation_search as new_cluster_correlation_search

"""
This module contains different clustering functions and can be extended to new ones.
All method signatures should look like this:
    def name(sG: SimulationGraph, params: dict) -> dict:
Each clustering function should only return a dictionary with label-node key-value pairs   
"""


def correlation_clustering(simulationGraph: BaseGraph, params: dict) -> dict:
    """
    This clustering implementation uses the clustering algorithm mentioned in:
        'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'

    Args:
        :param simulationGraph: SimulationGraph where to find clusters
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

    clusters = old_cluster_correlation_search(G=simulationGraph.get_nx_graph_copy(
        weights), s=s, max_attempts=max_attempts, max_iters=max_iters)
    # clusters = cluster_correlation_search(G=sG.get_nx_graph_with_hard_pos_neg_edges(), s=s, max_attempts=max_attempts, max_iters=max_iters)
    # clusters = cluster_correlation_search(G=sG.G, s=s, max_attempts=max_attempts, max_iters=max_iters)

    community_node = {}
    for cluster_id, cluster in enumerate(clusters):
        community_node[cluster_id] = list(cluster)

    return community_node


def new_correlation_clustering(simulationGraph: BaseGraph, params: dict) -> dict:
    """
    This clustering implementation uses the clustering algorithm mentioned in:
        'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'

    Args:
        :param simulationGraph: SimulationGraph where to find clusters
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
    assert type(max_iters) == int

    split_flag = params.get('split_flag', True)
    assert type(split_flag) == bool

    ru_old_cluster = params.get('ru_old_cluster', False)
    assert type(ru_old_cluster) == bool
    # ===Guard Phase===

    initial = simulationGraph.get_labels() if ru_old_cluster else []

    clusters, stats = new_cluster_correlation_search(G=simulationGraph.get_nx_graph_copy(
        weights), s=s, max_attempts=max_attempts, max_iters=max_iters, initial=initial, split_flag=split_flag)

    community_node = {}
    for cluster_id, cluster in enumerate(clusters):
        community_node[cluster_id] = list(cluster)

    return community_node
