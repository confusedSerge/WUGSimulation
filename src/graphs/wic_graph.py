import networkx as nx

from graphs.base_graph import BaseGraph

class WICGraph(BaseGraph):
    """
    A WIC Graph.

    TODO: This is mostly just an example and should be filled with needed logic
    """
    def __init__(self):
        super().__init__()
        
        self.last_node = None
        self.last_edge = None

    # Edge functionality
    def get_edge(self, u_node: int, v_node: int, **params) -> float:
        """
        Returns the weight of an edge between two nodes, if it is present.

        Args:
            :param u_node: first node
            :param v_node: second node
            :return float: weight
        """
        return self.G.graph['edge_weight'].get((u_node, v_node), None)

    def get_last_added_edge(self):
        raise self.last_edge

    def add_edge(self, node_u: int, node_v: int, weight: float, **params) -> None:
        raise NotImplementedError

    def add_edges(self, edge_list: list, **params) -> None:
        raise NotImplementedError

    # Node functionality
    def get_last_added_node(self) -> tuple:
        return self.last_node

    def get_num_added_edges(self) -> int:
        raise NotImplementedError

    # other
    def get_nx_graph_copy(self, weight: str) -> nx.Graph:
        """
        Creates a new nx.Graph with specified weight dict.

        Args:
            :param weight: which weight dict to use to populate the nx.Graphs edges
        """
        weights = self.G.graph.get(weight, None)
        assert type(weights) == dict

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
    
    def build_nx_graph_from_dicts(self, node_dict: dict, edge_dict: dict, weight: str, weight_modifier = lambda x: x):
        """
        Completes the nxgraph with the given parameters, which were created using the TODO

        Args:
            :param node_dict: Node Dictionary
            :param edge_dict: Edge Dictionary
            :param weight: which weights to use for the edges
            :param weight_modifier: lambda function modifying the weights of each edge
        """
        for k, v in node_dict.items():
            self.G.add_node(k, data=v)
        self.last_node = k

        for k, v in edge_dict.items():
            self.G.graph['edge_weight'][k] = weight_modifier(v[weight])
            self.G.add_edge(*k, weight=weight_modifier(v[weight]) ,data=v)
        self.last_edge = (*k, weight_modifier(v[weight]))

        self.labels = [-1] * len(node_dict)

    def __str__(self):
        return 'Number of Nodes: {}\nNumber of Edges: {}\nNumber of Communities: {}'\
            .format(self.get_number_nodes(), self.get_number_edges(), self.get_number_communities())
