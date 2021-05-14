import networkx as nx
import pickle

from graphs.base_graph import BaseGraph


class WUSimulationGraph(BaseGraph):

    def __init__(self, max_nodes: int):
        super().__init__()

        # Max number of nodes
        self._max_nodes = max_nodes

        # differently weighted edges. Need to be addressed through params in functions
        self.G.graph['edge_soft_weight'] = {} # only important for wug

        self.G.graph['distribution'] = 'simulated'  # only important for wug

    def get_edge(self, u_node: int, v_node: int) -> int:
        return self.G.graph['edge_weight'].get((u_node, v_node), None)

    def add_edge(self, node_u: int, node_v: int, weight: int) -> None:
        self.add_edges([(node_u, node_v, weight)])

    def add_edges(self, edge_list: list) -> None:
        assert len(edge_list) != 0
        for edge in edge_list:
            assert len(edge) == 3

        self.G.add_weighted_edges_from(edge_list)

        # update edge/weight dicts
        for u, v, w in edge_list:
            u, v = sorted([u, v])

            self.G.graph['edge_weight'][(u, v)] = w
            self.G.graph['edge_soft_weight'][(u, v)] = w - 2.5
            
            if self.G.graph['weight_edge'].get(w, None) == None:
                self.G.graph['weight_edge'][v] = []
            self.G.graph['weight_edge'][v].append((u, v))

    def update_community_nodes_membership(self, new_community_nodes: dict) -> None:
        assert type(new_community_nodes) == dict
        self.G.graph['community_nodes'] = new_community_nodes

        # -1 resembles that this node is not yet in the graph
        label=[-1] * self._max_nodes
        for k, v in self.G.graph['community_nodes'].items():
            for node in v:
                label[node]=k

    def __str__(self):
        return 'Distribution: {}\n  \
            Number of Nodes: {}\n   \
            Number of Edges: {}\n   \
            Number of Communities: {}'\
            .format(self.G.graph['distribution'], self.get_number_nodes(),
                    self.get_number_edges(), self.get_number_communities())
