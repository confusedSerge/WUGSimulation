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
        # should be replaced by node_community dict
        self.labels = []

        # community/node dict
        self.G.graph['community_nodes'] = {}
        self.G.graph['node_community'] = {}

        # edge/weight dicts (should be base weights)
        self.G.graph['edge_weight'] = {}
        self.G.graph['weight_edge'] = {}
        self.G.graph['edge_soft_weight'] = {}

        # edge weight history
        self.G.graph['edge_weight_history'] = {}

        # metric dict
        self.G.graph['metrics'] = {}

    # Edge functionality

    def get_edge(self, u_node: int, v_node: int, **params) -> float or None:
        u, v = sorted([u_node, v_node])
        edge = self.G.get_edge_data(u_node, v_node)
        return None if edge is None else edge['weight']

    def get_edge_weight_history(self, u_node: int, v_node: int, **params) -> list:
        return self.G.graph['edge_weight_history'].get((u_node, v_node), [])

    def get_last_added_edge(self):
        raise NotImplementedError

    def add_edge(self, node_u: int, node_v: int, weight: float, **params) -> None:
        self.G.add_weighted_edges_from([(node_u, node_v, weight)])

        u, v = sorted([node_u, node_v])
        self.G.graph['edge_weight'][(u, v)] = weight
        self.G.graph['edge_soft_weight'][(u, v)] = weight - 2.5

        if self.G.graph['weight_edge'].get(weight, None) is None:
            self.G.graph['weight_edge'][weight] = []
        self.G.graph['weight_edge'][weight].append((u, v))

        if self.G.graph['edge_weight_history'].get((u, v), None) is None:
            self.G.graph['edge_weight_history'][(u, v)] = []
        self.G.graph['edge_weight_history'][(u, v)].append(weight)

    def add_edges(self, edge_list: list, **params) -> None:
        for edge in edge_list:
            self.add_edge(*edge)

    # Node functionality
    def get_last_added_node(self) -> tuple:
        raise NotImplementedError

    def get_num_added_edges(self) -> int:
        raise NotImplementedError

    # info functions
    def get_number_nodes(self) -> int:
        return len(self.G.nodes)

    def get_number_edges(self) -> int:
        return len(self.G.edges)

    def get_community_of_node(self, node: int) -> int:
        return self.G.graph['node_community'][node]

    def get_number_communities(self) -> int:
        return len(self.G.graph['community_nodes'])

    def get_community_sizes(self) -> list:
        return [len(v) for k, v in self.G.graph['community_nodes'].items()]

    def get_dictionary_of_graph(self, name: str) -> list:
        return self.G.graph[name]

    def get_edge_weight(self) -> dict:
        return self.G.graph['edge_weight']

    def get_weight_edge(self) -> dict:
        return self.G.graph['weight_edge']

    def get_community_nodes(self) -> dict:
        return self.G.graph['community_nodes']

    def get_node_community(self) -> dict:
        return self.G.graph['node_community']

    def get_metric_dict(self) -> dict:
        return self.G.graph['metrics']

    def get_labels(self) -> list:
        return self.labels

    # util functions
    def update_community_nodes_membership(self, new_community_nodes: dict) -> None:
        assert type(new_community_nodes) == dict
        self.G.graph['community_nodes'] = new_community_nodes

        for k, v in new_community_nodes.items():
            for nodes in v:
                self.G.graph['node_community'][nodes] = k

    def get_nx_graph_copy(self, weight: str = 'edge_weight', fmap=lambda x: x) -> nx.Graph:
        weights = self.G.graph.get(weight, None)
        assert type(weights) == dict

        graph = nx.Graph()

        if weights is not None:
            graph.add_weighted_edges_from(list(map(lambda k: (*k[0], k[1]), weights.items())))
        else:
            assert callable(fmap)
            graph.add_weighted_edges_from(list(map(lambda k: (*k[0], fmap(k[1])), self.G.graph['edge_weight'].items())))

        return graph

    def add_new_weight_dict(self, name: str, weight_modifier: lambda x: x) -> None:
        self.G.graph[name] = {}
        for k, v in self.G.graph['edge_weight'].items():
            self.G.graph[name][k] = weight_modifier(v)

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
        return 'Number of Nodes: {}\nNumber of Edges: {}\nNumber of Communities: {}'\
            .format(self.get_number_nodes(), self.get_number_edges(), self.get_number_communities())
