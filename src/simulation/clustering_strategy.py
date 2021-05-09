from simulation.simulation_graph import SimulationGraph
from simulation.clustering_utils.cluster_correlation_search import cluster_correlation_search

"""
This module contains different clustering functions and can be extended to new ones.
All method signatures should look like this:
    def name(sG: SimulationGraph, params: dict) -> dict:
Each clustering function should only return a dictionary with label-node key-value pairs   
"""

def correlation_clustering(sG: SimulationGraph, params: dict) -> dict:
    """
    This clustering implementation uses the clustering algorithm mentioned in:
        'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'
    
    # TODO returns more clusters than s! Correct?
    Args:
        :param sG: SimulationGraph where to find clusters
        :param s: maximal number of senses a word can have
        :param max_attempts: number of restarts for optimization
        :param max_iters: number of iterations for optimization
        :return labels: dict with label-node key-value pairs  
    """
    s = params.get('s', 10)
    assert type(s) == int

    max_attempts = params.get('max_attempts', 200)
    assert type(max_attempts) == int

    max_iters = params.get('max_iters', 500)
    assert type(max_iters) == int

    clusters = cluster_correlation_search(G=sG.get_nx_graph_with_soft_pos_neg_edges(), s=s, max_attempts=max_attempts, max_iters=max_iters)
    # clusters = cluster_correlation_search(G=sG.get_nx_graph_with_hard_pos_neg_edges(), s=s, max_attempts=max_attempts, max_iters=max_iters)
    # clusters = cluster_correlation_search(G=sG.G, s=s, max_attempts=max_attempts, max_iters=max_iters)

    community_node = {}
    for cluster_id, cluster in enumerate(clusters):
        community_node[cluster_id] = list(cluster)

    return community_node