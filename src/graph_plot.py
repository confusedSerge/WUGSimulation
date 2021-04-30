import networkx as nx
import matplotlib.pyplot as plt
import graph_sim

class GraphDrawer:

    def __init__(self):
        self.G = nx.Graph()

        self.community_node_labels_dic = {}
        self.edge_weight_label_dic = {}
        self.weight_edge_dic = {}

        self._mapper_builder()

    def _mapper_builder(self):
        self.node_mapper_shape = {}
        for ids, label in enumerate(list('so^>v<dph8')):
            self.node_mapper_shape[ids] = label

        self.node_mapper_color = {}
        for ids, label in enumerate(list('bgrcmykw')):
            self.node_mapper_color[ids] = label

        self.edge_mapper_color = {}
        for ids, label in enumerate(['white', 'lightgray', 'darkgray', 'dimgray', 'black']):
            self.edge_mapper_color[ids] = label

    def build_new_nx_graph_from_graph_sim(self, graph: graph_sim.GraphSimulator):
        self.build_new_nx_graph(graph.G, graph.labels)

    def draw_graph_from_graph_sim(self, graph: graph_sim.GraphSimulator, edge_label_flag: bool = False, save_flag: bool = False, path: str = None):
        self.build_new_nx_graph_from_graph_sim(graph)
        self.draw_graph(edge_label_flag, save_flag, path)

    def build_new_nx_graph(self, adjacency_matrix, node_labels):
        """
        Builds a new Networkx Graph
        """
        # Builds weighted edge list, edge-weight dict, and weight-edges dict
        edge_list = []
        edge_weight_label_dic = {}
        weight_edge_dic = {}

        for i in range(len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix)):
                edge_list.append((i, j, int(adjacency_matrix[i][j])))
                edge_weight_label_dic[(i, j)] = int(adjacency_matrix[i][j])

                if weight_edge_dic.get(int(adjacency_matrix[i][j]), None) == None:
                    weight_edge_dic[int(adjacency_matrix[i][j])] = []
                weight_edge_dic[int(adjacency_matrix[i][j])].append((i, j))


        # init new graph
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(edge_list)

        # save edge/weight dicts
        self.edge_weight_label_dic = edge_weight_label_dic
        self.weight_edge_dic = weight_edge_dic

        # community-node dict
        community_node_labels_dic = {}
        for node_id, label in enumerate(node_labels):
            if community_node_labels_dic.get(label, None) == None:
                community_node_labels_dic[label] = []
            community_node_labels_dic[label].append(node_id)

        self.community_node_labels_dic = community_node_labels_dic

    def draw_graph(self, edge_label_flag: bool = False, save_flag: bool = False, path: str = None):
        #
        pos = nx.spring_layout(self.G, k=1/(len(self.G.nodes)))
        plt.figure(figsize=(20, 20))

        # just ehhhh
        options = {"node_size": 500, "alpha": 0.8}

        for k, v in self.community_node_labels_dic.items():
            nx.draw_networkx_nodes(
                self.G, pos, nodelist=v, node_color=self.node_mapper_color.get(k, 'w'), node_shape=self.node_mapper_shape.get(k, '8'), **options)

        for k, v in self.weight_edge_dic.items():
            nx.draw_networkx_edges(
                self.G, pos, edgelist=v, edge_color=self.edge_mapper_color.get(k, 'w'), label=k)

        if edge_label_flag:
            nx.draw_networkx_edge_labels(
                self.G, pos=pos, edge_labels=self.edge_weight_label_dic)

        if save_flag:
            if path == None:
                raise Exception("No name given")
            plt.savefig(path)
        else:
            plt.show()
