from graphs.base_graph import BaseGraph


def cluster_unclustered_nodes(graph: BaseGraph):
    clustered_nodes = set()
    for k, v in graph.get_community_nodes().items():
        for n in v:
            clustered_nodes.add(n)
    node_set = set(graph.G.nodes())
    return node_set - clustered_nodes


def sort_len_nodes(cluster2node: dict) -> dict:
    sorted_part = {}
    for i, (k, v) in enumerate(dict(sorted(cluster2node.items(), key=lambda x: len(x[1]), reverse=True)).items()):
        sorted_part[i] = list(sorted(v))
    return sorted_part


def louvain_cluster_sort(node2cluster: dict) -> dict:
    # inverting dict and putting nodes in same cluster together
    out = {}
    for k, v in node2cluster.items():
        _tmp = out.get(v, [])
        _tmp.append(k)
        out[v] = _tmp
    # final sort
    return sort_len_nodes(out)
