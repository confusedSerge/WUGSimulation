import networkx as nx
import pickle


class SimulationGraph():

    def __init__(self):
        self.G = nx.Graph(directed=False)

        # differently weighted edges
        self.G.graph['mvd_edges'] = []

        # community/node dict
        self.G.graph['community_node'] = {}

        # edge/weight dicts
        self.G.graph['edge_weight'] = {}
        self.G.graph['weight_edge'] = {}

        self.G.graph['number_nodes'] = 0
        self.G.graph['communities'] = 0
        self.G.graph['distribution'] = 'simulated'

    def add_edge(self, node_u: int, node_v: int, weight: int):
        self.add_edges([(node_u, node_v, weight)])

    def add_edges(self, edge_list: list):
        assert len(edge_list) != 0
        for edge in edge_list:
            assert len(edge) == 3

        self.G.add_weighted_edges_from(edge_list)
        self.G.graph['mvd_edges'].extend(
            list(map(lambda w: (w[0], w[1], w[2] - 2.5), edge_list)))

        # update edge/weight dicts
        for u, v, w in edge_list:
            u, v = sorted([u, v])
            self.G.graph['edge_weight'][(u, v)] = w

    def update_community_membership(self, community_node: dict):
        assert type(community_node) == dict
        self.G.graph['community_node'] = community_node
        self.G.graph['communities'] = len(self.G.graph['community_node'])

    def update_graph_attributes(self):
        for k, v in self.G.graph['edge_weight'].items():
            if self.G.graph['weight_edge'].get(v, None) == None:
                self.G.graph['weight_edge'][v] = []
            self.G.graph['weight_edge'][v].append(k)

        self.G.graph['number_nodes'] = len(self.G.nodes())

    def get_nx_graph_with_pos_neg_edges(self) -> nx.Graph:
        nx_graph = nx.Graph(directed=False)
        nx_graph.add_weighted_edges_from(self.G.graph['mvd_edges'])

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
