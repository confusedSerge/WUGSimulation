import pickle
from graphs.annotated_graph import AnnotatedGraph
from simulation.clustering.clustering_strategy import new_correlation_clustering
from visualization.graph_visualization import draw_graph_graphviz as draw

path = 'data/graphs/sim_graphs/dwug/sim_ks_logsoft/2021_07_09_15_25/final/dwug_n100_k5.graph'

with open(path, 'rb') as file:
    graph: AnnotatedGraph = pickle.load(file)
file.close()

cluster = new_correlation_clustering(graph, {'weights': 'edge_soft_weight'})
graph.update_community_nodes_membership(cluster)
draw(graph, "DWUG with k=1")

print(graph.get_num_added_edges())
