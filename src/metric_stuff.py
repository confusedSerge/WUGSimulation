import os
import pickle
from graphs.annotated_graph import AnnotatedGraph
from simulation.clustering.clustering_strategy import new_correlation_clustering
from visualization.graph_visualization import draw_graph_graphviz as draw

path_graphs_soft = 'data/graphs/sim_graphs/randomsampling/sim_ks_logsoft/2021_07_09_15_27/intermediate/k1'
path_graphs_hard = 'data/graphs/sim_graphs/randomsampling/sim_ks_loghard/2021_07_09_15_28/intermediate/k1'
path_out = 'data/graphs/sim_graphs/randomsampling/sim_ks_logsofthard'

def load_graphs(path: str):
    sort_func = lambda x: x.get_num_added_edges()
    graphs = []
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith('.graph'):
                with open('{}/{}'.format(path, file), 'rb') as fg:
                    graphs.append(pickle.load(fg))
                fg.close()

    # Check, that all loaded items are graphs
    for i, graph in enumerate(graphs):
        try:
            assert isinstance(graph, AnnotatedGraph)
        except AssertionError:
            graphs.pop(i)

    graphs.sort(key=sort_func)
    return graphs

graphs_soft = load_graphs(path_graphs_soft)
graphs_hard = load_graphs(path_graphs_hard)
print('ay')