import numpy as np

from graphs.base_graph import BaseGraph
from graphs.utils.util import generate_pdf_parameter_dicts

from graspologic.simulations import sbm


class FittedGraph(BaseGraph):

    def __init__(self, fitted_dict: dict) -> None:
        r""" Fitted Graph which will be build from the provided dict

        Parameters
        ----------
        fitted_dict : dict: containing communities, distribution, parameters for distribution
        """
        super().__init__()

        self.adjacency_matrix, self.labels, self.communities_probability = self._generate_graph_from_dict(fitted_dict)

        # self._clean_adj_matrix()

        self.adjacency_matrix += 1
        self._build_nx_graph_rep()

    def _build_nx_graph_rep(self) -> None:
        """
        """
        # ===Calculation Phase===
        # Builds weighted edge list, edge-weight dict, and weight-edges dict
        edge_list = []
        edge_weight = {}
        weight_edge = {}

        for i in range(len(self.adjacency_matrix)):
            for j in range(i + 1, len(self.adjacency_matrix)):
                edge_list.append((i, j, int(self.adjacency_matrix[i][j])))
                edge_weight[(i, j)] = int(self.adjacency_matrix[i][j])

                if weight_edge.get(int(self.adjacency_matrix[i][j]), None) is None:
                    weight_edge[int(self.adjacency_matrix[i][j])] = []
                weight_edge[int(self.adjacency_matrix[i][j])].append((i, j))

        # community-node dict
        community_nodes = {}
        node_community = {}
        for node_id, label in enumerate(self.labels):
            if community_nodes.get(label, None) is None:
                community_nodes[label] = []
            community_nodes[label].append(node_id)
            node_community[node_id] = label

        # ===Calculation Phase END===

        # ===Build Phase===
        self.G.add_weighted_edges_from(edge_list)

        # save edge/weight dicts
        self.G.graph['edge_weight'] = edge_weight
        self.G.graph['weight_edge'] = weight_edge

        # add rest of information
        self.G.graph['community_nodes'] = community_nodes
        self.G.graph['node_community'] = node_community
        # ===Build Phase===

    def _clean_adj_matrix(self):
        for ii in range(len(self.adjacency_matrix)):
            for jj in range(len(self.adjacency_matrix)):
                if jj != ii and self.adjacency_matrix[ii][jj] == 0:
                    self.adjacency_matrix[ii][jj] += 1

    def _generate_graph_from_dict(self, fitted_dict: dict):
        pdf = fitted_dict['pdf']
        parameters = fitted_dict['param']
        communities_probability = np.ones((fitted_dict['communities'], fitted_dict['communities']), dtype=int)

        return *sbm(n=fitted_dict['community_size'], p=communities_probability, wt=pdf, wtargs=parameters, return_labels=True), communities_probability
