import networkx as nx
import pickle

from graphs.base_graph import BaseGraph


class WUSimulationGraph(BaseGraph):

    def __init__(self, max_nodes: int):
        """
        Simulated WUG.
        Used for simulating the annotation process of word usages.

        Args:
            :param max_nodes: nodes in the sampled WUG
        """
        super().__init__()

        # Max number of nodes
        self._max_nodes = max_nodes
        self.labels = [-1] * self._max_nodes

        # differently weighted edges. Need to be addressed through params in functions
        self.G.graph['edge_soft_weight'] = {}  # only important for wug

        self.G.graph['distribution'] = 'simulated'  # only important for wug

    def get_edge(self, u_node: int, v_node: int) -> float:
        """
        Returns the weight of an edge between two nodes, if it is present.

        Args:
            :param u_node: first node
            :param v_node: second node
            :return float: weight
        """
        return self.G.graph['edge_weight'].get((u_node, v_node), None)

    def add_edge(self, node_u: int, node_v: int, weight: float) -> None:
        """
        Adds an edge to the graph.

        Args:
            :param u_node: first node
            :param v_node: second node
            :param weight: weight
        """
        self.add_edges([(node_u, node_v, weight)])

    def add_edges(self, edge_list: list) -> None:
        """
        Adds an list of edges to the graph.

        Expected input of list:
            [(node1: int, node2:int, weight: float)]

        Args:
            :param edge_list: edges to add
        """
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
                self.G.graph['weight_edge'][w] = []
            self.G.graph['weight_edge'][w].append((u, v))

    def get_nx_graph_copy(self, weight: str) -> nx.Graph:
        """
        Creates a new nx.Graph with specified weight dict.

        Args:
            :param weight: which weight dict to use to populate the nx.Graphs edges
        """
        weights = self.G.graph.get(weight, None)
        assert type(weights) != None and type(weights) == dict

        graph = nx.Graph()
        graph.add_weighted_edges_from(list(map(lambda k: (*k[0], k[1]), weights.items())))

        return graph

    def update_community_nodes_membership(self, new_community_nodes: dict) -> None:
        """
        Updates the node-community membership of this graph (which node belongs to which community/cluster).

        Args:
            :param new_community_nodes: new membership dict
        """
        assert type(new_community_nodes) == dict
        self.G.graph['community_nodes'] = new_community_nodes

        # -1 resembles that this node is not yet in the graph
        for k, v in self.G.graph['community_nodes'].items():
            for node in v:
                self.labels[node] = k

    def __str__(self):
        return 'Distribution: {}\nNumber of Nodes: {}\nNumber of Edges: {}\nNumber of Communities: {}'\
            .format(self.G.graph['distribution'], self.get_number_nodes(), self.get_number_edges(), self.get_number_communities())
