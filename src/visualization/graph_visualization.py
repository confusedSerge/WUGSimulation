import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from graphs.base_graph import BaseGraph


def draw_graph_graphviz(graph: BaseGraph, plot_title: str, edge_label_flag: bool = False, save_flag: bool = False, path: str = None) -> None:
    """
    Draws the Graph using graphviz as layout position information.

    Args:
        :param graph: Graph to draw
        :param title: title of plot
        :param edge_label_flag: if edge labels should be included
        :param save_flag: if plot should be saved
        :param path: where to save the plot
    """
    pos = nx.drawing.nx_agraph.graphviz_layout(graph.G)
    _draw_graph(graph, pos, plot_title, edge_label_flag, save_flag, path)


def draw_graph_spring(graph: BaseGraph, plot_title: str, edge_label_flag: bool = False, save_flag: bool = False, path: str = None) -> None:
    """
    Draws the Graph using spring as layout position information.

    Args:
        :param graph: Graph to draw
        :param title: title of plot
        :param edge_label_flag: if edge labels should be included
        :param save_flag: if plot should be saved
        :param path: where to save the plot
    """
    pos = nx.spring_layout(graph.G, scale=50)
    _draw_graph(graph, pos, plot_title, edge_label_flag, save_flag, path)


def _draw_graph(graph: BaseGraph, pos, plot_title: str, edge_label_flag: bool = False, save_flag: bool = False, path: str = None) -> None:
    px = 1 / plt.rcParams['figure.dpi']
    plt.figure(figsize=(800 * px, 800 * px))

    options = {"node_size": 250, "alpha": 0.8}

    for k, v in graph.get_community_nodes().items():
        nx.draw_networkx_nodes(
            graph.G, pos, nodelist=v, node_color=_get_node_color(k), node_shape=_get_node_shape(k), label=len(v), **options)

    for k, v in reversed(graph.get_weight_edge().items()):
        nx.draw_networkx_edges(
            graph.G, pos, edgelist=v, edge_color=_get_edge_color(int(k)))

    if edge_label_flag:
        nx.draw_networkx_edge_labels(
            graph.G, pos=pos, edge_labels=graph.get_edge_weight())

    plt.legend(title='Legend', scatterpoints=1)
    plt.figtext(.12, .02, str(graph))
    plt.title(plot_title)

    if save_flag:
        if path is None:
            raise NotADirectoryError("No name given")
        plt.savefig(path)
    else:
        plt.show()

    plt.clf()
    plt.close()


def _get_node_shape(i: int) -> str:
    return 'so^>v<dph8'[i % 10]


def _get_node_color(i: int) -> str:
    return 'bgrcmyk'[i % 7]


def _get_edge_color(i: int) -> str:
    return ['white', 'lightgray', 'darkgray', 'dimgray', 'black'][i % 5]
