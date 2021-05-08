from simulation.simulation_graph import SimulationGraph
from simulation.stopping_utils.utils import check_connectivity_two_clusters

"""
This module contains different stopping criterion functions and can be extended to new ones.
All method signatures should look like this:
    def name(sG: SimulationGraph, params: dict) -> bool:
Each stopping criterion function should only return a boolean, that indicates, if the criterion is reached  
"""

def cluster_connected(sG: SimulationGraph, params: dict) -> bool:
    """
    Cluster connectivity as described in paper:
     'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'

    This criterion looks, if all clusters with minimum size m are connected at least with n edges with every other cluster.
    If only 1 community exist, that is bigger than m, and its size is bigger than max_size_one_cluster, this function returns true.

    Args:
        :param sG: SimulationGraph to check on
        :param cluster_min_size: minimum size of cluster to consider, default = 5
        :param min_num_edges: minimum number of edges between clusters, 
            can be int for number of edges, or 'fully' for fully connected clusters; default = 1
        :param min_size_one_cluster: int at which this function considers the only cluster as big enough
    """
    cluster_min_size = params.get('cluster_min_size', 5)
    assert type(cluster_min_size) == int

    min_num_edges = params.get('min_num_edges', 1)
    assert type(min_num_edges) == int or min_num_edges == 'fully'

    min_size_one_cluster = params.get('min_size_one_cluster', len(sG.G.nodes()))
    assert type(min_size_one_cluster) == int

    # communities = dict(map(lambda kv: (kv[0], kv[1]) if len(kv[1]) >= 10 else None, sG.self.G.graph['communities'].items()))
    communities = []
    for k, v in sG.self.G.graph['communities'].items():
        if len(v) >= cluster_min_size:
            communities.append(v)

    if len(communities) == 0:
        return False
        
    if len(communities) == 1:
        return len(communities[0]) >= min_size_one_cluster

    flag = True

    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            min_connections = min_num_edges if min_num_edges != 'fully' else len(communities[i]) * len(communities[j]) 
            flag = flag and check_connectivity_two_clusters(sG.G.edges(), communities[i], communities[j], min_num_edges)

    return flag