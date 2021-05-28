from graphs.base_graph import BaseGraph
from simulation.utils.stopping_utils import check_connectivity_two_clusters

"""
This module contains different stopping criterion functions and can be extended to new ones.
All method signatures should look like this:
    def name(sG: SimulationGraph, params: dict) -> bool:
Each stopping criterion function should only return a boolean, that indicates, if the criterion is reached  
"""


def cluster_connected(simulationGraph: BaseGraph, params: dict) -> bool:
    """
    Cluster connectivity as described in paper:
     'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'

    This criterion looks, if all clusters with minimum size m are connected at least with n edges with every other cluster.
    If only 1 community exist, that is bigger than m, and its size is bigger than max_size_one_cluster, this function returns true.

    Args:
        :param simulationGraph: Simulation Graph to check on
        :param cluster_min_size: minimum size of cluster to consider, default = 5
        :param min_num_edges: minimum number of edges between clusters, 
            can be int for number of edges, or 'fully' for fully connected clusters; default = 1
        :param min_size_one_cluster: int at which this function considers the only cluster as big enough
        :return flag: if stopping criterion is met
    """
    # ===Guard Phase===
    cluster_min_size = params.get('cluster_min_size', 5)
    assert type(cluster_min_size) == int

    min_num_edges = params.get('min_num_edges', 1)
    assert type(min_num_edges) == int or min_num_edges == 'fully'

    min_size_one_cluster = params.get(
        'min_size_one_cluster', simulationGraph.get_number_nodes())
    assert type(min_size_one_cluster) == int

    # communities = dict(map(lambda kv: (kv[0], kv[1]) if len(kv[1]) >= 10 else None, sG.self.G.graph['communities'].items()))
    communities = []
    for k, v in simulationGraph.get_community_nodes().items():
        if len(v) >= cluster_min_size:
            communities.append(v)
    # ===Guard Phase End===

    if len(communities) == 0:
        return False

    if len(communities) == 1:
        return len(communities[0]) >= min_size_one_cluster

    flag = True

    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            min_connections = min_num_edges if min_num_edges != 'fully' else len(
                communities[i]) * len(communities[j])
            flag = flag and check_connectivity_two_clusters(
                simulationGraph.G.edges(), communities[i], communities[j], min_connections)

    return flag


def number_edges_found(simulationGraph: BaseGraph, params: dict) -> bool:
    """
    Rather simple stopping criterion, where if an number of edges has been discovered,
    this returns true.
    This means, if the simulation graph contains at least n edges, it returns true, else false.
    Args:
        :param simulationGraph: Simulation Graph to check on
        :param number_edges: number of edges that sG should contain
        :return flag: if sG contains the number of edges
    """
    number_edges = params.get('number_edges', None)
    assert type(number_edges) == int

    return simulationGraph.get_number_edges() >= number_edges


def percentage_edges_found(simulationGraph: BaseGraph, params: dict) -> bool:
    """
    Rather simple stopping criterion, where if the simulation graph contains a certaint percentage of edges
    this returns true.
    Args:
        :param simulationGraph: SimulationGraph to check on
        :param percentage: percentage of edges that sG should contain
        :param number_edges: number of max edges
        :return flag: if sG contains the percentage of edges
    """
    percentage = params.get('percentage', None)
    assert type(percentage) == float

    number_edges = params.get('number_edges', None)
    assert type(number_edges) == int

    return simulationGraph.get_number_edges() >= (percentage * number_edges)

def edges_added(simulationGraph: BaseGraph, params: dict) -> bool:
    """
    Rather simple stopping criterion, where it checks how many edges were added to the simulation graph.
    Important, duplicate are also included
    Args:
        :param simulationGraph: SimulationGraph to check on
        :param num_edges: number of max edges added
        :return flag: if sG contains the percentage of edges
    """
    number_edges = params.get('number_edges', None)
    assert type(number_edges) == int

    return simulationGraph.get_num_added_edges() >= number_edges
