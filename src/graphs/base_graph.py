import pickle
import networkx as nx

class BaseGraph():

    def __init__(self):
        self.G = nx.Graph(directed=False)
        self.labels = []

        # community/node dict
        self.G.graph['community_nodes'] = {}

        # edge/weight dicts
        self.G.graph['edge_weight'] = {}
        self.G.graph['weight_edge'] = {}


    # Edge functionality
    def get_edge(self, u_node: int, v_node: int) -> int:
        raise NotImplementedError

    def add_edge(self, node_u: int, node_v: int, weight: int) -> None:
        raise NotImplementedError

    def add_edges(self, edge_list: list) -> None:
        raise NotImplementedError

    # info functions
    def get_number_nodes(self) -> int:
        return len(self.G.nodes)

    def get_number_edges(self) -> int:
        return len(self.G.edges)

    def get_number_communities(self) -> int:
        return len(self.G.graph['community_nodes'])

    def get_community_sizes(self) -> int:
        return [len(v) for k, v in self.G.graph['community_nodes'].items()]

    # util functions
    def update_community_nodes_membership(self, new_community_nodes: dict) -> None:
        raise NotImplementedError

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

    def __str__(self):
        raise NotImplementedError()