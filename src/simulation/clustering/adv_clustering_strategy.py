from graphs.base_graph import BaseGraph
from simulation.clustering.utils.new_cluster_correlation_search import cluster_correlation_search
from simulation.clustering.utils.utils import cluster_unclustered_nodes

"""
This module provides a more advance use of clustering, like using time step information.
If any information is being used, a reset function has to be provided
"""

_current_run = 0

def time_degrading_clustering(graph: BaseGraph, params: dict) -> dict:
    """
    This clustering implementation uses the clustering algorithm mentioned in:
        'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning',
        but extends it, by using time information. 
        At the 0-th step a full clustering is performed. After this each following clustering
        will use the previous clustering and degrade max_attempts and max_iters by a factor 2.
        This is done n times, after which a full clustering is performed again.

    Args:
        :param graph: Graph to be clustered
        :param weights: weights to be used for clustering
        :param s: maximal number of senses a word can have
        :param max_attempts: number of restarts for optimization
        :param max_iters: number of iterations for optimization
        :param max_degrade: number of iterations till full clustering
        :param split_flag: optional flag, if non evidence cluster should be splitted
        :return labels: dict with label-node key-value pairs  
    """
    # ===Guard Phase===
    weights = params.get('weights', 'edge_weight')
    assert type(weights) == str
    
    s = params.get('s', 10)
    assert type(s) == int

    max_attempts = params.get('max_attempts', 200)
    assert type(max_attempts) == int

    max_iters = params.get('max_iters', 500)
    assert type(max_iters) == int    
    
    max_degrade = params.get('max_degrade', 5)
    assert type(max_degrade) == int
    max_degrade += 1

    split_flag = params.get('split_flag', True)
    assert type(split_flag) == bool

    global _current_run
    # ===Guard Phase===

    initial = [set(v) for k, v in sorted(graph.get_community_nodes().items())] if _current_run else []

    unknown_cluster = cluster_unclustered_nodes(graph) if _current_run else set()
    if len(unknown_cluster) > 0:
        initial.append(unknown_cluster) 

    clusters, stats = cluster_correlation_search(G=graph.get_nx_graph_copy(weights), s=s, max_attempts=int(max_attempts/2**_current_run), max_iters=int(max_iters/2**_current_run), initial=initial, split_flag=split_flag)

    community_node = {}
    for cluster_id, cluster in enumerate(clusters):
        community_node[cluster_id] = list(cluster)

    _current_run = (_current_run + 1) % max_degrade

    return community_node

def time_degrading_clustering_reset():
    global _current_run
    _current_run = 0