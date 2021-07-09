import pickle
from graphs.annotated_graph import AnnotatedGraph
from simulation.clustering.clustering_strategy import new_correlation_clustering
from visualization.graph_visualization import draw_graph_graphviz as draw

path = 'data/graphs/sim_graphs/randomsampling/sim_ks_logsoft/2021_07_09_15_00/final/randomsampling_n100_k3.graph'

with open(path, 'rb') as file:
    graph: AnnotatedGraph = pickle.load(file)
file.close()

cluster = new_correlation_clustering(graph, {'weights': 'edge_soft_weight'})
graph.update_community_nodes_membership(cluster)
# draw(graph, "RS with k=1")

# print(graph.get_num_added_edges())

print(graph.get_community_nodes())
b = set() 
for k, v in graph.get_community_nodes().items():
    for n in v:
        b.add(n)
a = set(graph.G.nodes())

print(a - b)