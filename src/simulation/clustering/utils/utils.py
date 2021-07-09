from graphs.base_graph import BaseGraph

def cluster_unclustered_nodes(graph: BaseGraph):
    clustered_nodes = set()
    for k, v in graph.get_community_nodes().items():
        for n in v:
            clustered_nodes.add(n)
    node_set = set(graph.G.nodes())
    return node_set - clustered_nodes