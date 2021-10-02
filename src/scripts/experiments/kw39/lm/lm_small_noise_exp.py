from graphs.simulation_graph import SimulationGraph
from graphs.utils.distribution import Binomial

from simulation.clustering.clustering_strategy import louvain_method_clustering as lm

from visualization.graph_visualization import draw_graph_gt as draw

# without noise
graph = SimulationGraph([100], None, Binomial(3, 1.0, 1))
draw(graph)

cl = lm(graph, {})
graph.update_community_nodes_membership(cl)
draw(graph)

# with noise!
graph = SimulationGraph([100], None, Binomial(3, 0.9, 1))
draw(graph)

cl = lm(graph, {})
graph.update_community_nodes_membership(cl)
draw(graph)
