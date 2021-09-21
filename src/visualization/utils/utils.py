import numpy as np
import graph_tool
import networkx as nx

from graphs.base_graph import BaseGraph


def generate_graphtool_graph(graph: BaseGraph):
    position_information = _calc_positional_information(graph.G.copy())
    graph_tool_graph, nx2gt_vertex_id, gt2nx_vertex_id = _nxgraph_to_graphtoolgraph(graph.G.copy(), position_information, graph.get_node_community())

    return graph_tool_graph, nx2gt_vertex_id, gt2nx_vertex_id


def _calc_positional_information(graph: nx.Graph):
    edges_negative = [(i, j) for (i, j) in graph.edges() if graph[i][j]['weight'] < 2.5]
    graph.remove_edges_from(edges_negative)

    return nx.nx_agraph.graphviz_layout(graph, prog='sfdp')


def _nxgraph_to_graphtoolgraph(graph: nx.Graph, position_dict: dict, community: dict):
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

    vertex_position = graph_tool_graph.new_vertex_property('vector<double>')
    for i, (k, v) in enumerate(position_dict.items()):
        vertex_list = [vertex for vertex in graph_tool_graph.get_vertices() if graph_tool_graph.vp.id[graph_tool_graph.vertex(vertex)] == k][0]
        vertex_position[graph_tool_graph.vertex(vertex_list)] = v
    graph_tool_graph.vp.pos = vertex_position

    return graph_tool_graph, nx2gt_vertex_id, gt2nx_vertex_id
