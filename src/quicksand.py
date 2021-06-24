import os
import pickle
from analysis.metrics import invers_entropy_distance

path_true = 'data/graphs/true_graphs/k_c_var/2021_06_11_10_35/true_graph_wug_n500_k3_clog_iter_0.9_5_dbinomial_3_0.99_anone.graph'
with open(path_true, 'rb') as file:
    true_graph = pickle.load(file)
file.close()

path_graph = 'data/graphs/sim_graphs/pagerank/sim_ks_logsoft/2021_06_11_10_54/intermediate/k3'
graphs = []
for _, _, files in os.walk(path_graph):
    for file in files:
        if file.endswith('.graph'):
            with open('{}/{}'.format(path_graph, file), 'rb') as fg:
                graphs.append(pickle.load(fg))
            fg.close()


# def sort_func(x): return x.get_num_added_edges()
# graphs.sort(key=sort_func)

for graph in graphs:
    print(graph.get_num_added_edges(),
          invers_entropy_distance(true_graph, graph, params={'threshold': 2.5}))
