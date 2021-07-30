import pickle
import numpy as np
from simulation.clustering.clustering_strategy import new_correlation_clustering 
from simulation.clustering.clustering_strategy import connected_components_clustering
from simulation.clustering.clustering_strategy import chinese_whisper_clustering
from simulation.clustering.clustering_strategy import louvain_method_clustering

from graphs.base_graph import BaseGraph
from visualization.graph_visualization import draw_graph_graphviz as draw

path = 'data/graphs/kw29/bigdata/true/n100_k{}_log{}_0.graph'