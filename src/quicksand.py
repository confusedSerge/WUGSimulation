import numpy as np
import random
import pickle
from itertools import combinations

from graphs.base_graph import BaseGraph
from graphs.annotated_graph import AnnotatedGraph

from simulation.sampling.sampling_strategy import multiple_edges_randomwalk
from simulation.clustering.clustering_strategy import chinese_whisper_clustering

from visualization.graph_visualization import draw_graph_graphviz as draw

# basegraph
basegraph = BaseGraph()
edges = combinations([*range(100)], 2)

for edge in edges:
    basegraph.add_edge(*edge, 4)

clusters = chinese_whisper_clustering(basegraph, {'weights': 'edge_soft_weight'})
basegraph.update_community_nodes_membership(clusters)

draw(basegraph, 'K5')

# sampling
ann = AnnotatedGraph(basegraph.get_number_nodes())

for ii in range(100):
    edge_list = multiple_edges_randomwalk(basegraph, {'rounds': 2, 'start': ann.get_last_added_node, 'sample_per_node': 5})
    ann.add_edges(edge_list)

    clusters = chinese_whisper_clustering(ann, {'weights': 'edge_soft_weight'})
    ann.update_community_nodes_membership(clusters)

    draw(ann, 'Step {}'.format(ii))
