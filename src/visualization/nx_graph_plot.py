import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from graphs.wu_graph import WUGraph

class GraphDrawer:
    # TODO: Should probably not be a class

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
        pos = nx.spring_layout(G, scale=50)
        # pos = nx.spring_layout(G, k=1/(len(G.nodes)), scale=20)
        
        px = 1/plt.rcParams['figure.dpi'] 
        plt.figure(figsize=(800*px, 800*px))

        # just ehhhh
        options = {"node_size": 250, "alpha": 0.8}

        for k, v in G.graph['community_node'].items():
            nx.draw_networkx_nodes(
                G, pos, nodelist=v, node_color=self.node_mapper_color[k % len(self.node_mapper_color)], node_shape=self.node_mapper_shape[k % len(self.node_mapper_shape)], label=len(v), **options)

        for k, v in reversed(G.graph['weight_edge'].items()):
            nx.draw_networkx_edges(
                G, pos, edgelist=v, edge_color=self.edge_mapper_color.get(k, 'w'))

        if edge_label_flag:
            nx.draw_networkx_edge_labels(
                G, pos=pos, edge_labels=G.graph['edge_weight'])

        plt.legend(title='Legend', scatterpoints=1)
        plt.figtext(.12, .02, 'Distribution: {}\nNumber of Nodes: {}\nNumber of Edges: {}\nNumber of Communities: {}'.format(G.graph['distribution'], G.graph['number_nodes'], G.graph['number_edges'], G.graph['communities']))


        if save_flag:
            if path == None:
                raise Exception("No name given")
            plt.savefig(path)
        else:
            plt.show()
