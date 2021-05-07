import networkx as nx
import matplotlib.pyplot as plt

from true_graph.true_graph import TrueGraph

class GraphDrawer:

    def __init__(self):
        self.community_node_labels_dic = {}
        self.edge_weight_label_dic = {}
        self.weight_edge_dic = {}

        self._mapper_builder()

    def _mapper_builder(self):
        self.node_mapper_shape = {}
        for ids, label in enumerate(list('so^>v<dph8')):
            self.node_mapper_shape[ids] = label

        self.node_mapper_color = {}
        for ids, label in enumerate(list('bgrcmyk')):
            self.node_mapper_color[ids] = label

        self.edge_mapper_color = {}
        for ids, label in enumerate(['white', 'lightgray', 'darkgray', 'dimgray', 'black']):
            self.edge_mapper_color[ids] = label

    def draw_graph(self, G: nx.Graph, edge_label_flag: bool = False, save_flag: bool = False, path: str = None):
        #
        pos = nx.spring_layout(G, scale=20)
        # pos = nx.spring_layout(G, k=1/(len(G.nodes)), scale=20)
        plt.figure(figsize=(200, 200))

        # just ehhhh
        options = {"node_size": 250, "alpha": 0.8}

        for k, v in G.graph['community_node_labels_dic'].items():
            nx.draw_networkx_nodes(
                G, pos, nodelist=v, node_color=self.node_mapper_color[k % len(self.node_mapper_color)], node_shape=self.node_mapper_shape[k % len(self.node_mapper_shape)], **options)

        for k, v in reversed(G.graph['weight_edge_dic'].items()):
            nx.draw_networkx_edges(
                G, pos, edgelist=v, edge_color=self.edge_mapper_color.get(k, 'w'), label=k)

        if edge_label_flag:
            nx.draw_networkx_edge_labels(
                G, pos=pos, edge_labels=G.graph['edge_weight_label_dic'])

        if save_flag:
            if path == None:
                raise Exception("No name given")
            plt.savefig(path)
        else:
            plt.show()
