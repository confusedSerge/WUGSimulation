import networkx as nx
import pickle


class SimulationGraph():

    def __init__(self):
        self.G = nx.Graph(directed=False)
        self.labels = []

        # differently weighted edges
        self.G.graph['soft_mvd_edges'] = []
        self.G.graph['hard_mvd_edges'] = []

        # community/node dict
        self.G.graph['community_node'] = {}

        # edge/weight dicts
        self.G.graph['edge_weight'] = {}
        self.G.graph['weight_edge'] = {}

        self.G.graph['number_nodes'] = 0
        self.G.graph['number_edges'] = 0
        self.G.graph['communities'] = 0
        self.G.graph['community_sizes'] = []
        self.G.graph['distribution'] = 'simulated'

    def add_edge(self, node_u: int, node_v: int, weight: int):
        self.add_edges([(node_u, node_v, weight)])

    def add_edges(self, edge_list: list):
        assert len(edge_list) != 0
        for edge in edge_list:
            assert len(edge) == 3

        self.G.add_weighted_edges_from(edge_list)
        
        self.G.graph['soft_mvd_edges'].extend(
            list(map(lambda w: (w[0], w[1], w[2] - 2.5), edge_list)))
        self.G.graph['hard_mvd_edges'].extend(
            list(map(lambda w: (w[0], w[1], (w[2] - 2.5) * 2), edge_list)))

        # update edge/weight dicts
        for u, v, w in edge_list:
            u, v = sorted([u, v])
            self.G.graph['edge_weight'][(u, v)] = w

        self.G.graph['number_edges'] = len(self.G.edges)

    def update_community_membership(self, community_node: dict):
        assert type(community_node) == dict
        self.G.graph['community_node'] = community_node
        self.G.graph['communities'] = len(self.G.graph['community_node'])
        self.G.graph['community_sizes'] = [(com_id, len(v)) for com_id, v in self.G.graph['community_node'].items()]


    def update_graph_attributes(self):
        for k, v in self.G.graph['edge_weight'].items():
            if self.G.graph['weight_edge'].get(v, None) == None:
                self.G.graph['weight_edge'][v] = []
            self.G.graph['weight_edge'][v].append(k)

        self.G.graph['number_nodes'] = len(self.G.nodes())
        self.G.graph['number_edges'] = len(self.G.edges())

        self.G.graph['communities'] = len(self.G.graph['community_node'])
        self.G.graph['community_sizes'] = [(com_id, len(v)) for com_id, v in self.G.graph['community_node'].items()]

    def get_nx_graph_with_soft_pos_neg_edges(self) -> nx.Graph:
        nx_graph = nx.Graph(directed=False)
        nx_graph.add_weighted_edges_from(self.G.graph['soft_mvd_edges'])

        return nx_graph

    def get_nx_graph_with_hard_pos_neg_edges(self) -> nx.Graph:
        nx_graph = nx.Graph(directed=False)
        nx_graph.add_weighted_edges_from(self.G.graph['hard_mvd_edges'])

        return nx_graph

    def get_label_list(self, size: int) -> list:
        label = [-1]*size
        for k, v in self.G.graph['community_node'].items():
            for node in v:
                label[node] = k
        for i in range(len(label)):
            if label[i] == -1:
                k + 1
                label[i] = k

        return label

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
        return "number_nodes: {} \ncommunities: {} \ncommunity_sizes: {}".format(
            self.G.graph['number_nodes'], self.G.graph['communities'],
            [(com_id, len(v)) for com_id, v in self.G.graph['community_node'].items()])
