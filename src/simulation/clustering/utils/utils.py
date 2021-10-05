import numpy as np
import networkx as nx

import graph_tool
from graph_tool.inference import minimize_blockmodel_dl
from graph_tool.inference.blockmodel import BlockState

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


def generate_graphtool_graph(graph: BaseGraph):
    graph_tool_graph, nx2gt_vertex_id, gt2nx_vertex_id = _nxgraph_to_graphtoolgraph(graph.G.copy(), graph.get_node_community())

    return graph_tool_graph, nx2gt_vertex_id, gt2nx_vertex_id


def _nxgraph_to_graphtoolgraph(graph: nx.Graph, community: dict):
    graph_tool_graph = graph_tool.Graph(directed=False)

    nx2gt_vertex_id = dict()
    gt2nx_vertex_id = dict()
    for i, node in enumerate(graph.nodes()):
        nx2gt_vertex_id[node] = i
        gt2nx_vertex_id[i] = node

    new_weights = []
    for i, j in graph.edges():
        current_weight = graph[i][j]['weight']
        if current_weight != 0 and not np.isnan(current_weight):
            graph_tool_graph.add_edge(nx2gt_vertex_id[i], nx2gt_vertex_id[j])
            new_weights.append(current_weight)

    original_edge_weights = graph_tool_graph.new_edge_property("double")
    original_edge_weights.a = new_weights
    graph_tool_graph.ep['weight'] = original_edge_weights

    new_vertex_id = graph_tool_graph.new_vertex_property('int')
    for k, v in nx2gt_vertex_id.items():
        new_vertex_id[v] = k
    graph_tool_graph.vp.id = new_vertex_id

    communities = graph_tool_graph.new_vertex_property('int')
    for k, v in community.items():
        communities[nx2gt_vertex_id[k]] = v
    graph_tool_graph.vp['community'] = communities

    return graph_tool_graph, nx2gt_vertex_id, gt2nx_vertex_id


def minimize(graph: graph_tool.Graph, distribution="discrete-binomial", deg_corr=False) -> BlockState:
    return minimize_blockmodel_dl(graph,
                                  state_args=dict(deg_corr=deg_corr, recs=[graph.ep.weight], rec_types=[distribution]),
                                  multilevel_mcmc_args=dict(B_min=1, B_max=30, niter=100, entropy_args=dict(adjacency=False, degree_dl=False)))
