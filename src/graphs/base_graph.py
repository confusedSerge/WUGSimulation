import pickle
import networkx as nx

class BaseGraph():
    """
    Base Graph for this framework.
    All graph-classes should inherit from this class, 
        as this class implements the bare minimum functionality,
        that a graphs needs to work with the current framework.
    Thus also allowing for new graphs to be used with this framework! 
    """

    def __init__(self):
        self.G = nx.Graph(directed=False)
        self.labels = []

        # community/node dict
        self.G.graph['community_nodes'] = {}

        # edge/weight dicts
        self.G.graph['edge_weight'] = {}
        self.G.graph['weight_edge'] = {}

        # metric dict
        self.G.graph['metrics'] = {}


    # Edge functionality
    def get_edge(self, u_node: int, v_node: int, **params) -> float:
        raise NotImplementedError

    def get_last_added_edge(self):
        raise NotImplementedError

    def add_edge(self, node_u: int, node_v: int, weight: float, **params) -> None:
        raise NotImplementedError

    def add_edges(self, edge_list: list, **params) -> None:
        raise NotImplementedError

    # Node functionality
    def get_last_added_node(self):
        raise NotImplementedError

    # info functions
    def get_number_nodes(self) -> int:
        return len(self.G.nodes)

    def get_number_edges(self) -> int:
        return len(self.G.edges)

    def get_number_communities(self) -> int:
        return len(self.G.graph['community_nodes'])

    def get_community_sizes(self) -> list:
        return [len(v) for k, v in self.G.graph['community_nodes'].items()]

    def get_edge_weight(self) -> dict: 
        return self.G.graph['edge_weight']

    def get_weight_edge(self) -> dict: 
        return self.G.graph['weight_edge']

    def get_community_nodes(self) -> dict: 
        return self.G.graph['community_nodes']

    def get_metric_dict(self) -> dict:
        return self.G.graph['metrics']

    # util functions
    def update_community_nodes_membership(self, new_community_nodes: dict) -> None:
        raise NotImplementedError

    def get_nx_graph_copy(self, weight: str) -> nx.Graph:
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