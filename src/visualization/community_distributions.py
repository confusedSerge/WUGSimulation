import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

# TODO Clean me please
 
def community_distribution(G: nx.Graph):
    communities = G.graph['community_node']
    edges = G.edges
    edge_weigt = G.graph['edge_weight']

    in_communities = _calc_distribution_in_communities(
        communities, edges, edge_weigt)
    between_communities = _calc_distribution_between_communities(
        communities, edges, edge_weigt)

    img_names = []

    for k, v in in_communities.items():
        title = "Community {} - Nodes: {} - Edges: {}".format(
            k, v['nodes'], v['edges'])
        observed_values = [v[i] for i in range(1, 5)]

        plot_values_oneside(observed_values, True, title, "data/figs/tmp/{}".format(k))
        img_names.append("{}".format(k))

    for k, v in between_communities.items():
        title = "Community '{}-{}' - Edges: {}".format(*k, v['edges'])
        observed_values = [v[i] for i in range(1, 5)]

        plot_values_oneside(observed_values, False, title, "data/figs/tmp/{}-{}".format(*k))
        img_names.append("{}-{}".format(*k))

    combine_community_plots(img_names, len(communities), "detailed_distribution.png", "data/figs/tmp/")



def _calc_distribution_in_communities(communities, edges, edge_weight) -> dict:
    in_communities = {}
    for community, nodes in communities.items():
        com_dict = {'nodes': len(nodes), 'edges': 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for u in range(len(nodes)):
            for v in range(u + 1, len(nodes)):
                node_u, node_v = sorted([nodes[u], nodes[v]])
                if (node_u, node_v) in edges:
                    com_dict['edges'] = com_dict['edges'] + 1
                    com_dict[edge_weight[(node_u, node_v)]] = com_dict[edge_weight[(
                        node_u, node_v)]] + 1
        for i in range(1, 5):
            com_dict[i] = com_dict[i] / com_dict['edges']
        in_communities[community] = com_dict
    return in_communities


def _calc_distribution_between_communities(communities, edges, edge_weight):
    between_communities = {}
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            between_communities[(i, j)] = _calc_distribution_between_two_communities(
                communities[i], communities[j], edges, edge_weight)
    return between_communities


def _calc_distribution_between_two_communities(com_1, com_2, edges, edge_weight):
    com_dict = {'edges': 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for node_u in com_1:
        for node_v in com_2:
            node_u, node_v = sorted([node_u, node_v])
            if (node_u, node_v) in edges:
                com_dict['edges'] = com_dict['edges'] + 1
                com_dict[edge_weight[(node_u, node_v)]
                         ] = com_dict[edge_weight[(node_u, node_v)]] + 1
    for i in range(1, 5):
        com_dict[i] = com_dict[i] / com_dict['edges']
    return com_dict


def plot_values_oneside(observed_v, in_community, title, path=""):
    fig, ax = plt.subplots(figsize=(14, 7))
    color_inside_observed = (0.411, 0.674, 0.909, 0.9)
    color_outside_observed = (0.933, 0.588, 0.623, 0.8)
    color_inside_inferred = (0.027, 0.419, 0.772)
    color_outside_inferred = (0.866, 0.031, 0.117)

    prelabel = title

    plt.bar(
        [1, 2, 3, 4],
        height=observed_v,
        width=1,
        color=color_inside_observed if in_community else color_outside_observed,
        label="Observed",
        tick_label=[1, 2, 3, 4]
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


def combine_community_plots(imgs, community_count, title, path=""):

    blank_image = None

    for img_name in imgs:
        img = PIL.Image.open(path + img_name + ".png")

        if(len(img_name) == 1):
            x = int(img_name) * img.width
            y = int(img_name) * img.height
        else:
            parts = img_name.split('-')
            x = int(parts[0]) * img.width
            y = int(parts[1]) * img.height

        if blank_image == None:
            w = community_count * img.width
            h = community_count * img.height

            blank_image = PIL.Image.new("RGB", (w, h), color=(255, 255, 255))

        blank_image.paste(img, (x,y))
 
    blank_image.save(path + title)
