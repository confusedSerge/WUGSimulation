import pickle

from graphs.annotated_graph import AnnotatedGraph
from visualization.graph_visualization import draw_graph_gt as draw_graph
from simulation.clustering.clustering_strategy import chinese_whisper_clustering as cw
from simulation.clustering.clustering_strategy import sbm_clustering

path = 'data/graphs/kw33/sim/modifiedrandomwalk/n100_k10_log0.9-4-modifiedrandomwalk_cw/n100_k10_log0.9-4-modifiedrandomwalk_cw_j5000.graph'
with open(path, 'rb') as file:
    graph: AnnotatedGraph = pickle.load(file)
file.close()

cl = cw(graph, {'weights': 'edge_soft_weight'})
print(cl)
graph.update_community_nodes_membership(cl)
draw_graph(graph, 'CW_Clustering.png')

cl = sbm_clustering(graph, {})
print(cl)
graph.update_community_nodes_membership(cl)
draw_graph(graph, 'SBM_Clustering.png')
