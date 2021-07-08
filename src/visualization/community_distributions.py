import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from graphs.base_graph import BaseGraph

# This is an unholy abomination

def community_distribution(graph: BaseGraph, name: str, directory: str, combine: bool = False) -> None:
    """
    Creates edge weight distribution plots for between and inside the communities of a graph.

    Args:
        :param graph: Graph on which to base the plots 
        :param name: name of the plots 
        :param directory: where plots will be saved 
        :param combine: if plots should also be combined to one img containing all plots 
    """
    assert isinstance(graph, BaseGraph)

    # ===Find communities===
    in_communities = _calc_distribution_in_communities(
        graph.get_community_nodes(), graph.G.edges(), graph.get_edge_weight())
    between_communities = _calc_distribution_between_communities(
        graph.get_community_nodes(), graph.G.edges(), graph.get_edge_weight())

    # ===Find what bins===
    bins = set()

    for v in in_communities.values():
        num_nodes, num_edges, observed_values = v
        bins = bins.union(set(observed_values.keys()))


    for v in between_communities.values():
        num_nodes, num_edges, observed_values = v
        bins = bins.union(set(observed_values.keys()))

    # ===Plot community distribution=== 
    img_names = []
    for k, v in in_communities.items():
        num_nodes, num_edges, observed_values = v
        title = "Community {} - Nodes: {} - Edges: {}".format(k, num_nodes, num_edges)

        for _bin in bins:
            if observed_values.get(_bin, None) == None:
                observed_values[_bin] = 0

        _plot_values_oneside(observed_values, True, title, "{}/{}_{}".format(directory, name, k))
        img_names.append(("{}_{}".format(name, k), "{}".format(k)))

    for k, v in between_communities.items():
        num_nodes, num_edges, observed_values = v
        title = "Community '{}-{}' - Edges: {}".format(*k, num_edges)

        for _bin in bins:
            if observed_values.get(_bin, None) == None:
                observed_values[_bin] = 0

        _plot_values_oneside(observed_values, False, title, "{}/{}_{}-{}".format(directory, name, *k))
        img_names.append(("{}_{}-{}".format(name, *k), "{}-{}".format(*k)))

    # ===combine community distribution=== 
    if combine:
        _combine_community_plots(img_names, graph.get_number_communities(), "{}_combined_dist.png".format(name), directory)


def _calc_distribution_in_communities(communities: dict, edges: list, edge_weight: dict) -> dict:
    in_communities = {}
    for community, nodes in communities.items():
        num_edges = 0
        weight_num_edge = {}
        for u in range(len(nodes)):
            for v in range(u + 1, len(nodes)):
                node_u, node_v = sorted([nodes[u], nodes[v]])
                if (node_u, node_v) in edges:
                    num_edges += 1

                    if weight_num_edge.get(edge_weight[(node_u, node_v)], None) == None:
                        weight_num_edge[edge_weight[(node_u, node_v)]] = 0

                    weight_num_edge[edge_weight[(node_u, node_v)]] = weight_num_edge[edge_weight[(node_u, node_v)]] + 1

        for k, v in weight_num_edge.items():
            weight_num_edge[k] = v / num_edges
        in_communities[community] = (len(nodes), num_edges, weight_num_edge)
    return in_communities


def _calc_distribution_between_communities(communities: dict, edges: list, edge_weight: dict) -> dict:
    between_communities = {}
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            between_communities[(i, j)] = _calc_distribution_between_two_communities(
                communities[i], communities[j], edges, edge_weight)
    return between_communities


def _calc_distribution_between_two_communities(first_community: list, second_community: list, edges: list, edge_weight: dict) -> (int, int, dict):
    num_edges = 0
    weight_num_edge = {}

    for node_u in first_community:
        for node_v in second_community:
            node_u, node_v = sorted([node_u, node_v])
            if (node_u, node_v) in edges:
                num_edges += 1

                if weight_num_edge.get(edge_weight[(node_u, node_v)], None) == None:
                    weight_num_edge[edge_weight[(node_u, node_v)]] = 0

                weight_num_edge[edge_weight[(
                    node_u, node_v)]] = weight_num_edge[edge_weight[(node_u, node_v)]] + 1

    for k, v in weight_num_edge.items():
        weight_num_edge[k] = v / num_edges
    return (len(first_community) + len(second_community), num_edges, weight_num_edge)


def _plot_values_oneside(observed_values: dict, in_community: bool, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    color_inside_observed = (0.411, 0.674, 0.909, 0.9)
    color_outside_observed = (0.933, 0.588, 0.623, 0.8)

    plt.bar(
        list(observed_values.keys()),
        height=observed_values.values(),
        width=1,
        color=color_inside_observed if in_community else color_outside_observed,
        label="Observed",
        tick_label=list(observed_values.keys())
    )

    ax.set_title(title, fontsize=22)

    ax.set_ylim((0, 1))

    plt.legend(fontsize=20)
    plt.ylabel('Probability density', fontsize=22)
    plt.xlabel('Weights', fontsize=22)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.savefig(fname=path, dpi=150)
    plt.close(fig)


def _combine_community_plots(img_list: list, community_count: int, title: str, directory: str) -> None:

    blank_image = None

    for img_name, img_prop in img_list:
        img = PIL.Image.open("{}/{}.png".format(directory, img_name))

        if(len(img_prop) == 1):
            x = int(img_prop) * img.width
            y = int(img_prop) * img.height
        else:
            parts = img_prop.split('-')
            x = int(parts[0]) * img.width
            y = int(parts[1]) * img.height

        if blank_image == None:
            w = community_count * img.width
            h = community_count * img.height

            blank_image = PIL.Image.new("RGB", (w, h), color=(255, 255, 255))

        blank_image.paste(img, (x, y))

    blank_image.save("{}/{}".format(directory, title))
