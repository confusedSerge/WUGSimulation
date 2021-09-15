from graphs.annotated_graph import AnnotatedGraph
from simulation.clustering.clustering_strategy import chinese_whisper_clustering as cw
from visualization.graph_visualization import draw_graph_graphviz as draw

annotated_graph = AnnotatedGraph(100)
annotated_graph.add_edges([[0, 1, 4], [0, 1, 1], [1, 2, 4], [2, 0, 4], [2, 3, 4]])

cl = cw(annotated_graph, {'weights': 'edge_soft_weight'})
annotated_graph.update_community_nodes_membership(cl)

# draw(annotated_graph, 'Example', edge_label_flag=True)

annotated_graph.add_edges([[0, 1, 1], [1, 2, 1], [2, 0, 1]])

cl = cw(annotated_graph, {'weights': 'edge_soft_weight'})
annotated_graph.update_community_nodes_membership(cl)

# draw(annotated_graph, 'Example', edge_label_flag=True)

annotated_graph.add_edges([[0, 1, 4], [1, 2, 1], [2, 0, 1]])

cl = cw(annotated_graph, {'weights': 'edge_soft_weight'})
annotated_graph.update_community_nodes_membership(cl)

print(annotated_graph.G.degree())
print(len(annotated_graph.G.edges))
print(annotated_graph.G.graph.get('bla', None))
draw(annotated_graph, 'Example', edge_label_flag=True)
