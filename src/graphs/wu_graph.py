import pickle
import numpy as np
import networkx as nx

from graphs.base_graph import BaseGraph
from graphs.distribution import Distribution
from graspologic.simulations import sbm


class WUGraph(BaseGraph):

    def __init__(self, communities: int, communities_probability: list = None, distribution: Distribution = None):
        """
        WUG class.

        Args:
            :param communities: number of communities/word usages
            :param communities_probability: probability of a connection inside/between community
            :param distribution: used for creating weights between edges
        """
        super().__init__()

        self.adjacency_matrix, self.labels, self.communities_probability = self._gen_graph_from_params(
            communities, communities_probability, distribution)
        self.adjacency_matrix += 1

        self._build_nx_graph_rep()

        # adding distribution info
        self.G.graph['distribution'] = distribution

    def get_edge(self, u_node: int, v_node: int) -> int:
        return int(self.adjacency_matrix[u_node][v_node])

    def _build_nx_graph_rep(self) -> None:
        """
        """
        # ===Caclulation Phase===
        # Builds weighted edge list, edge-weight dict, and weight-edges dict
        edge_list = []
        edge_weight = {}
        weight_edge = {}

        for i in range(len(self.adjacency_matrix)):
            for j in range(i + 1, len(self.adjacency_matrix)):
                edge_list.append((i, j, int(self.adjacency_matrix[i][j])))
                edge_weight[(i, j)] = int(self.adjacency_matrix[i][j])

                if weight_edge.get(int(self.adjacency_matrix[i][j]), None) == None:
                    weight_edge[int(self.adjacency_matrix[i][j])] = []
                weight_edge[int(self.adjacency_matrix[i][j])].append((i, j))

        # community-node dict
        community_nodes = {}
        for node_id, label in enumerate(self.labels):
            if community_nodes.get(label, None) == None:
                community_nodes[label] = []
            community_nodes[label].append(node_id)
        # ===Caclulation Phase END===

        # ===Build Phase===
        self.G.add_weighted_edges_from(edge_list)

        # save edge/weight dicts
        self.G.graph['edge_weight'] = edge_weight
        self.G.graph['weight_edge'] = weight_edge

        # add rest of information
        self.G.graph['community_nodes'] = community_nodes
        # ===Build Phase===

    def _gen_graph_from_params(self, communities: list, communities_probability: list = None, distribution: Distribution = None):
        # ===Guard Phase===
        if distribution == None:
            raise AssertionError

        if communities_probability == None:
            communities_probability = np.ones(
                (len(communities), len(communities)), np.int)
        # ===Guard Phase End===

        return *sbm(n=communities, p=communities_probability, wt=distribution.get_distribution(
        ), wtargs=distribution.get_dist_param_dict(), return_labels=True), communities_probability

    def __str__(self):
        return 'Distribution: {}\nNumber of Nodes: {}\nNumber of Edges: {}\nNumber of Communities: {}'\
            .format(self.G.graph['distribution'], self.get_number_nodes(), self.get_number_edges(), self.get_number_communities())
