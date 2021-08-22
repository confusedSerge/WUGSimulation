import os
import numpy as np
import networkx as nx
import pickle

from datetime import datetime
from graphs.base_graph import BaseGraph
from analysis.metric_runner import MetricRunner
import analysis.metrics as am
import analysis.comparison_metrics as acm
import analysis.utils.utils as util
from simulation.clustering.clustering_strategy import new_correlation_clustering

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")
metric_runner = MetricRunner(name='Random Walk', info=current_time)
# Adding simple metrics
metric_runner.add_simple_metric(am.entropy_clustered, {})\
    .add_simple_metric(am.entropy_approximation, {})\
    .add_simple_metric(am.apd, {'sample_size': 500})\
    .add_simple_metric(am.hpd, {'sample_size': 500})

#stripped H and jsd
metric_runner.add_comparison_metric(acm.stripped_entropy, {})\
    .add_comparison_metric(acm.jensen_shannon_divergence, {})\
    .add_comparison_metric(acm.jsd_approximation_entropy, {})\
    .add_comparison_metric(acm.jsd_approximation_kld, {})

# diff metrics H(t) vs ...
metric_runner.add_comparison_metric(acm.distance_h_h, {})\
    .add_comparison_metric(acm.distance_h_hn, {})\
    .add_comparison_metric(acm.distance_h_apd, {'sample_size': 500})\
    .add_comparison_metric(acm.distance_h_hpd, {'sample_size': 500})

# diff metrics H(a) vs ...
metric_runner.add_comparison_on_self_metric(acm.distance_h_hn, {})\
    .add_comparison_on_self_metric(acm.distance_h_apd, {'sample_size': 500})\
    .add_comparison_on_self_metric(acm.distance_h_hpd, {'sample_size': 500})

# diff metrics H*(t, a) vs ...
metric_runner.add_comparison_metric(acm.distance_stripped_h_h, {})\
    .add_comparison_metric(acm.distance_stripped_h_hn, {})\
    .add_comparison_metric(acm.distance_stripped_h_apd, {'sample_size': 500})\
    .add_comparison_metric(acm.distance_stripped_h_hpd, {'sample_size': 500})


# Load graphs to use
true_graphs = [
    [
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k1_clog_iter_0.9_5_dbinomial_3_0.99.graph']*15,
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_13/true_graph_wug_n100_k3_clog_iter_0.1_5_dbinomial_3_0.99.graph']*15,
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_13/true_graph_wug_n100_k5_clog_iter_0.1_5_dbinomial_3_0.99.graph']*15,
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_13/true_graph_wug_n100_k10_clog_iter_0.1_5_dbinomial_3_0.99.graph']*15
    ], [
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k1_clog_iter_0.9_5_dbinomial_3_0.99.graph']*15,
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k3_clog_iter_0.9_5_dbinomial_3_0.99.graph']*15,
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k5_clog_iter_0.9_5_dbinomial_3_0.99.graph']*15,
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k10_clog_iter_0.9_5_dbinomial_3_0.99.graph']*15
    ]
]
ann_graphs = [
    [
        ['data/graphs/sim_graphs/randomwalk/sim_ks_loghard/2021_07_09_15_18/intermediate/k1'],
        ['data/graphs/sim_graphs/randomwalk/sim_ks_loghard/2021_07_09_15_18/intermediate/k3'],
        ['data/graphs/sim_graphs/randomwalk/sim_ks_loghard/2021_07_09_15_18/intermediate/k5'],
        ['data/graphs/sim_graphs/randomwalk/sim_ks_loghard/2021_07_09_15_18/intermediate/k10']
    ], [
        ['data/graphs/sim_graphs/randomwalk/sim_ks_logsoft/2021_07_09_15_17/intermediate/k1'],
        ['data/graphs/sim_graphs/randomwalk/sim_ks_logsoft/2021_07_09_15_17/intermediate/k3'],
        ['data/graphs/sim_graphs/randomwalk/sim_ks_logsoft/2021_07_09_15_17/intermediate/k5'],
        ['data/graphs/sim_graphs/randomwalk/sim_ks_logsoft/2021_07_09_15_17/intermediate/k10']
    ]
]

true_graphs = util.load_graph_from_path_file(true_graphs)
ann_graphs = util.load_graph_from_path_dir(ann_graphs, '.graph', sort_func = lambda x: x.get_num_added_edges(), shape=(2, 4, 15))
print('Loaded Graphs')
# clustering for simulated graphs, so that metrics can be calculated
sanity = np.zeros(ann_graphs.shape)
it = np.nditer(np.zeros(sanity.shape), flags=['multi_index'])
for _ in it:
    _cl = new_correlation_clustering(ann_graphs[it.multi_index], {'weights': 'edge_soft_weight'})
    ann_graphs[it.multi_index].update_community_nodes_membership(_cl)
print('Clustered Graphs')

metric_runner.run(ann_graphs, true_graphs)
metric_result = metric_runner.metric_result
print('Calced Metrics')

out_metric = 'data/graphs/sim_graphs/randomwalk/sim_ks_logsofthard/{}'.format(current_time)
os.makedirs(out_metric)

with open('{}/metric.data'.format(out_metric), 'wb') as file:
    pickle.dump(metric_result, file)
file.close()