import pickle
import numpy as np
import networkx as nx

from true_graph.distribution import *
from graspologic.simulations import sbm


# TODO typing
# TODO good documentation

class TrueGraph:

    def __init__(self, communities: list, communities_probability: list = None, distribution: Distribution = None):
        # adjacency matrix
        self.G = None
        # labels[node] = community
        self.labels = None

        # #nodes
        self.number_nodes = sum(communities)
        # community distribution of nodes
        self.communities = communities
        # distribution used to generate communities/weights
        self.distribution = distribution

        self.G, self.labels, self.communities_probability = self._gen_graph_from_params(
            communities, communities_probability, distribution)
        self.G += 1

    def sample_edge(self, u_node: int, v_node: int) -> int:
        return int(self.G[u_node][v_node])

    def get_nx_graph_rep(self):
        """
        Returns the True Graph as a Networkx Graph
        """
        # Builds weighted edge list, edge-weight dict, and weight-edges dict
        edge_list = []
        edge_weight = {}
        weight_edge = {}

        for i in range(len(self.G)):
            for j in range(i + 1, len(self.G)):
                edge_list.append((i, j, int(self.G[i][j])))
                edge_weight[(i, j)] = int(self.G[i][j])

                if weight_edge.get(int(self.G[i][j]), None) == None:
                    weight_edge[int(self.G[i][j])] = []
                weight_edge[int(self.G[i][j])].append((i, j))

        # init new graph
        nx_graph = nx.Graph(directed=False)
        nx_graph.add_weighted_edges_from(edge_list)

        # save edge/weight dicts
        nx_graph.graph['edge_weight'] = edge_weight
        nx_graph.graph['weight_edge'] = weight_edge

        # community-node dict
        community_node = {}
        for node_id, label in enumerate(self.labels):
            if community_node.get(label, None) == None:
                community_node[label] = []
            community_node[label].append(node_id)

        nx_graph.graph['community_node'] = community_node

        # add rest of information
        nx_graph.graph['number_nodes'] = self.number_nodes
        nx_graph.graph['communities'] = self.communities
        nx_graph.graph['distribution'] = self.distribution

        return nx_graph

    def save_graph(self, path: str):
        with open(path, "xb") as file:
            pickle.dump(self, file)
        file.close()

    @staticmethod
    def load_graph(path: str):
        with open(path, "rb") as file:
            graph = pickle.load(file)
        file.close()
        return graph

    def _gen_graph_from_params(self, communities: list, communities_probability: list = None, distribution: Distribution = None):
        if distribution == None:
            raise AssertionError

        if communities_probability == None:
            communities_probability = np.ones(
                (len(communities), len(communities)), np.int)

        return *sbm(n=communities, p=communities_probability, wt=distribution.get_distribution(
        ), wtargs=distribution.get_dist_param_dict(), return_labels=True), communities_probability

    def __str__(self):
        return 'Adjacency Matrix {}, Labels {}'.format(self.G, self.labels)
