import os
import pickle
import numpy as np
import analysis.utils.utils as util
from analysis.metric_results import MetricResults
from analysis.metrics import entropy_clustered
from analysis.metrics import entropy_clustered_normalized
import visualization.metric_vis as vis

"""
Keys:
'stripped_entropy'
'entropy_clustered'
'entropy_approximation'
'apd'
'hpd'

'stripped_entropy_normalized'
'entropy_clustered_normalized'
'entropy_approximation_normalized'
'apd_normalized'
'hpd_normalized'

'jensen_shannon_divergence'

'jsd_approximation_entropy'
'jsd_approximation_kld'
'jsd_approximation_apd'
'jsd_approximation_hpd'

'jsd_approximation_entropy_normalized'
'jsd_approximation_kld_normalized'
'jsd_approximation_apd_normalized'
'jsd_approximation_hpd_normalized'
"""

with open('data/graphs/sim_graphs/randomsampling/metric_norm_run/island/2021_07_22_15_44/metric.data', 'rb') as file:
    metric_wi : MetricResults = pickle.load(file)
    short_wi = 'rs_is'
    long_wi = 'RandomSampling with Islands'
file.close()

with open('data/graphs/sim_graphs/randomsampling/metric_norm_run/wo_island/2021_07_22_15_44/metric.data', 'rb') as file:
    metric_woi : MetricResults = pickle.load(file)
    short_woi = 'rs_wo'
    long_woi = 'RandomSampling without Islands'
file.close()

true_graphs = [
    [
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k1_clog_iter_0.9_5_dbinomial_3_0.99.graph'],
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_13/true_graph_wug_n100_k3_clog_iter_0.1_5_dbinomial_3_0.99.graph'],
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_13/true_graph_wug_n100_k5_clog_iter_0.1_5_dbinomial_3_0.99.graph'],
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_13/true_graph_wug_n100_k10_clog_iter_0.1_5_dbinomial_3_0.99.graph']
    ], [
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k1_clog_iter_0.9_5_dbinomial_3_0.99.graph'],
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k3_clog_iter_0.9_5_dbinomial_3_0.99.graph'],
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k5_clog_iter_0.9_5_dbinomial_3_0.99.graph'],
        ['data/graphs/true_graphs/k_c_var/2021_07_09_13_14/true_graph_wug_n100_k10_clog_iter_0.9_5_dbinomial_3_0.99.graph']
    ]
]

true_graphs = util.load_graph_from_path_file(true_graphs)
steps = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
ks = [1, 3, 5, 10]
logs = ['0.1', '0.9']

# # with islands 
# out_path = 'data/figs/entropy_kw29/hs/{}'.format(short_wi)
# os.makedirs(out_path)
# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         base = [entropy_clustered(true_graphs[i][j][0], {})]*15
#         out_name = 'log{}_k{}.png'.format(log, k)
#         vis.line_ploter(steps, 'Entropy, {}, log={}, k={}'.format(long_wi, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path='{}/{}'.format(out_path, out_name), y_lim=(0, 6),
#             base=base,
#             hs=metric_wi.get_values('stripped_entropy', i, j),
#             h_ann=metric_wi.get_values('entropy_clustered', i, j),
#             hn_ann=metric_wi.get_values('entropy_approximation', i, j),
#             apd_ann=metric_wi.get_values('apd', i, j),
#             hpd_ann=metric_wi.get_values('hpd', i, j)
#             )

# # with islands normalized
# out_path = 'data/figs/entropy_kw29/hs/{}_norm'.format(short_wi)
# os.makedirs(out_path)
# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         base = [entropy_clustered_normalized(true_graphs[i][j][0], {})]*15
#         out_name = 'log{}_k{}.png'.format(log, k)
#         vis.line_ploter(steps, 'Entropy Normalized, {}, log={}, k={}'.format(long_wi, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path='{}/{}'.format(out_path, out_name), y_lim=(0, 1),
#             base=base,
#             hs=metric_wi.get_values('stripped_entropy_normalized', i, j),
#             h_ann=metric_wi.get_values('entropy_clustered_normalized', i, j),
#             hn_ann=metric_wi.get_values('entropy_approximation_normalized', i, j),
#             apd_ann=metric_wi.get_values('apd_normalized', i, j),
#             hpd_ann=metric_wi.get_values('hpd_normalized', i, j)
#             )


# # without islands 
# out_path = 'data/figs/entropy_kw29/hs/{}'.format(short_woi)
# os.makedirs(out_path)
# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         base = [entropy_clustered(true_graphs[i][j][0], {})]*15
#         out_name = 'log{}_k{}.png'.format(log, k)
#         vis.line_ploter(steps, 'Entropy, {}, log={}, k={}'.format(long_woi, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path='{}/{}'.format(out_path, out_name), y_lim=(0, 6),
#             base=base,
#             hs=metric_woi.get_values('stripped_entropy', i, j),
#             h_ann=metric_woi.get_values('entropy_clustered', i, j),
#             hn_ann=metric_woi.get_values('entropy_approximation', i, j),
#             apd_ann=metric_woi.get_values('apd', i, j),
#             hpd_ann=metric_woi.get_values('hpd', i, j)
#             )

# # without islands normalized
# out_path = 'data/figs/entropy_kw29/hs/{}_norm'.format(short_woi)
# os.makedirs(out_path)
# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         base = [entropy_clustered_normalized(true_graphs[i][j][0], {})]*15
#         out_name = 'log{}_k{}.png'.format(log, k)
#         vis.line_ploter(steps, 'Entropy Normalized, {}, log={}, k={}'.format(long_woi, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path='{}/{}'.format(out_path, out_name), y_lim=(0, 1),
#             base=base,
#             hs=metric_woi.get_values('stripped_entropy_normalized', i, j),
#             h_ann=metric_woi.get_values('entropy_clustered_normalized', i, j),
#             hn_ann=metric_woi.get_values('entropy_approximation_normalized', i, j),
#             apd_ann=metric_woi.get_values('apd_normalized', i, j),
#             hpd_ann=metric_woi.get_values('hpd_normalized', i, j)
#             )

# # with islands 
# out_path = 'data/figs/entropy_kw29/jsd/{}'.format(short_wi)
# os.makedirs(out_path)
# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         out_name = 'log{}_k{}.png'.format(log, k)
#         vis.line_ploter(steps, 'JSD, {}, log={}, k={}'.format(long_wi, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path='{}/{}'.format(out_path, out_name),
#             jsd=metric_wi.get_values('jensen_shannon_divergence', i, j),
#             jsd_h=metric_wi.get_values('jsd_approximation_entropy', i, j),
#             jsd_kld=metric_wi.get_values('jsd_approximation_kld', i, j),
#             jsd_apd=metric_wi.get_values('jsd_approximation_apd', i, j),
#             jsd_hpd=metric_wi.get_values('jsd_approximation_hpd', i, j)
#             )

# with islands normalized
out_path = 'data/figs/entropy_kw29/jsd/{}_norm_no_kld'.format(short_wi)
os.makedirs(out_path)
for i, log in enumerate(logs):
    for j, k in enumerate(ks):
        out_name = 'log{}_k{}.png'.format(log, k)
        vis.line_ploter(steps, 'JSD Normalized, {}, log={}, k={}'.format(long_wi, log, k), 'Judgement Points', 'Entropy',
            save_flag=True, save_path='{}/{}'.format(out_path, out_name),
            jsd=metric_wi.get_values('jensen_shannon_divergence', i, j),
            jsd_h=metric_wi.get_values('jsd_approximation_entropy_normalized', i, j),
            jsd_apd=metric_wi.get_values('jsd_approximation_apd_normalized', i, j),
            jsd_hpd=metric_wi.get_values('jsd_approximation_hpd_normalized', i, j)
            )

# # without islands 
# out_path = 'data/figs/entropy_kw29/jsd/{}'.format(short_woi)
# os.makedirs(out_path)
# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         out_name = 'log{}_k{}.png'.format(log, k)
#         vis.line_ploter(steps, 'JSD, {}, log={}, k={}'.format(long_woi, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path='{}/{}'.format(out_path, out_name),
#             jsd=metric_woi.get_values('jensen_shannon_divergence', i, j),
#             jsd_h=metric_woi.get_values('jsd_approximation_entropy', i, j),
#             jsd_kld=metric_woi.get_values('jsd_approximation_kld', i, j),
#             jsd_apd=metric_woi.get_values('jsd_approximation_apd', i, j),
#             jsd_hpd=metric_woi.get_values('jsd_approximation_hpd', i, j)
#             )

# without islands normalized
out_path = 'data/figs/entropy_kw29/jsd/{}_norm_no_kld'.format(short_woi)
os.makedirs(out_path)
for i, log in enumerate(logs):
    for j, k in enumerate(ks):
        out_name = 'log{}_k{}.png'.format(log, k)
        vis.line_ploter(steps, 'JSD Normalized, {}, log={}, k={}'.format(long_woi, log, k), 'Judgement Points', 'Entropy',
            save_flag=True, save_path='{}/{}'.format(out_path, out_name),
            jsd=metric_woi.get_values('jensen_shannon_divergence', i, j),
            jsd_h=metric_woi.get_values('jsd_approximation_entropy_normalized', i, j),
            jsd_apd=metric_woi.get_values('jsd_approximation_apd_normalized', i, j),
            jsd_hpd=metric_woi.get_values('jsd_approximation_hpd_normalized', i, j)
            )
