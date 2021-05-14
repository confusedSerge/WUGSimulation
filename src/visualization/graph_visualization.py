import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from graphs.base_graph import BaseGraph

def draw_graph_graphviz(baseGraph: BaseGraph, edge_label_flag: bool = False, save_flag: bool = False, path: str = None) -> None:
    """
    Draws the Graph using graphviz as layout position information.

    Args:
        :param baseGraph: Graph to draw
        :param edge_label_flag: if edge labels should be included
        :param save_flag: if plot should be saved
        :param path: where to save the plot
    """
    pos = nx.drawing.nx_agraph.graphviz_layout(baseGraph.G)
    _draw_graph(baseGraph, pos, edge_label_flag, save_flag, path)

def draw_graph_spring(baseGraph: BaseGraph, edge_label_flag: bool = False, save_flag: bool = False, path: str = None) -> None:
    """
    Draws the Graph using spring as layout position information.

    Args:
        :param baseGraph: Graph to draw
        :param edge_label_flag: if edge labels should be included
        :param save_flag: if plot should be saved
        :param path: where to save the plot
    """
    pos = nx.spring_layout(baseGraph.G, scale=50)
    _draw_graph(baseGraph, pos, edge_label_flag, save_flag, path)

def _draw_graph(baseGraph: BaseGraph, pos, edge_label_flag: bool = False, save_flag: bool = False, path: str = None) -> None:
    pos = nx.spring_layout(baseGraph.G, scale=50)
    # pos = nx.spring_layout(G, k=1/(len(G.nodes)), scale=20)
    
    px = 1/plt.rcParams['figure.dpi'] 
    plt.figure(figsize=(800*px, 800*px))

    # just ehhhh
    options = {"node_size": 250, "alpha": 0.8}

    for k, v in baseGraph.get_community_nodes().items():
        nx.draw_networkx_nodes(
            baseGraph.G, pos, nodelist=v, node_color=_get_node_color(k), node_shape=_get_node_shape(k), label=len(v), **options)

    for k, v in reversed(baseGraph.get_weight_edge().items()):
        nx.draw_networkx_edges(
            baseGraph.G, pos, edgelist=v, edge_color=_get_edge_color(k))

    if edge_label_flag:
        nx.draw_networkx_edge_labels(
            baseGraph.G, pos=pos, edge_labels=baseGraph.get_edge_weight())

    plt.legend(title='Legend', scatterpoints=1)
    plt.figtext(.12, .02, str(baseGraph))


    if save_flag:
        if path == None:
            raise Exception("No name given")
        plt.savefig(path)
    else:
        plt.show()


def _get_node_shape(i: int) -> str:
    return 'so^>v<dph8'[i % 10]

def _get_node_color(i: int) -> str:
    return 'bgrcmyk'[i % 7]

def _get_edge_color(i: int) -> str:
    return ['white', 'lightgray', 'darkgray', 'dimgray', 'black'][i % 5]

