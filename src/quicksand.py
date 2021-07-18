import numpy as np
import networkx as nx
import pickle

from graphs.base_graph import BaseGraph
from analysis.comparison_metrics import jensen_shannon_divergence
from analysis.comparison_metrics import jsd_approximation_entropy
from analysis.comparison_metrics import jsd_approximation_kld
from simulation.clustering.clustering_strategy import new_correlation_clustering

with open('data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k5_clog_iter_0.9_5_dbinomial_3_0.99.graph', 'rb') as fg:
    a: BaseGraph = pickle.load(fg)
fg.close()

with open('data/graphs/sim_graphs/randomsampling/sim_ks_logsoft/2021_07_09_15_27/intermediate/k5/randomsampling_n100_k5_500.graph', 'rb') as fg:
    b: BaseGraph = pickle.load(fg)
fg.close()

b_cl = new_correlation_clustering(b, {'weights': 'edge_soft_weight'})
b.update_community_nodes_membership(b_cl)

print('JSD')
print(jensen_shannon_divergence(a, b, {}))
print(jsd_approximation_entropy(a, b, {}))
print(jsd_approximation_kld(a, b, {}))