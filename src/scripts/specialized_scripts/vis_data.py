import pickle
import numpy as np
import analysis.utils.utils as util
from analysis.metric_results import MetricResults
from analysis.metrics import entropy_clustered
import visualization.metric_vis as vis

"""
Keys:
    entropy_clustered
    entropy_approximation
    apd
    hpd
    stripped_entropy

    jensen_shannon_divergence
    jsd_approximation_entropy
    jsd_approximation_kld

    distance_h_h
    distance_h_hn
    distance_h_apd
    distance_h_hpd
    
    distance_stripped_h_h
    distance_stripped_h_hn
    distance_stripped_h_apd
    distance_stripped_h_hpd
"""
with open('data/graphs/sim_graphs/dwug/sim_ks_logsofthard/2021_07_18_14_11/metric.data', 'rb') as file:
    metric : MetricResults = pickle.load(file)
    short = 'dwug'
    long = 'DWUG'
file.close()

# with open('data/graphs/sim_graphs/pagerank/sim_ks_logsofthard/2021_07_18_14_06/metric.data', 'rb') as file:
#     metric : MetricResults = pickle.load(file)
#     short = 'pr'
#     long = 'PageRank'
# file.close()

# with open('data/graphs/sim_graphs/randomwalk/sim_ks_logsofthard/2021_07_18_14_05/metric.data', 'rb') as file:
#     metric : MetricResults = pickle.load(file)
#     short = 'rw'
#     long = 'RandomWalk'
# file.close()

# with open('data/graphs/sim_graphs/randomsampling/sim_ks_logsofthard/2021_07_18_13_37/metric.data', 'rb') as file:
#     metric : MetricResults = pickle.load(file)
#     short = 'rs'
#     long = 'RandomSampling'
# file.close()

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
# for k, v in metric.metric_dict.items():
#     print(k)

steps = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
ks = [1, 3, 5, 10]
logs = ['0.1', '0.9']

# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         base = [entropy_clustered(true_graphs[i][j][0], {})]*15
#         out_path = 'data/figs/entropy/lineplots_hs/{}/log{}_k{}.png'.format(short, log, k)
#         vis.line_ploter(steps, 'Entropy Diff Annotated Graph, {}, log={}, k={}'.format(long, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path=out_path, y_lim=(0, 6),
#             base=base,
#             hs=metric.get_values('stripped_entropy', i, j),
#             h_ann=metric.get_values('entropy_clustered', i, j),
#             hn_ann=metric.get_values('entropy_approximation', i, j),
#             apd_ann=metric.get_values('apd', i, j),
#             hpd_ann=metric.get_values('hpd', i, j)
#             )

# hn = metricrs.get_values('distance_h_hn').flatten()
# hn = np.append(hn, metricrw.get_values('distance_h_hn').flatten())
# hn = np.append(hn, metricpr.get_values('distance_h_hn').flatten())

# apd = metricrs.get_values('distance_h_apd').flatten()
# apd = np.append(apd, metricrw.get_values('distance_h_apd').flatten())
# apd = np.append(apd, metricpr.get_values('distance_h_apd').flatten())

# hpd = metricrs.get_values('distance_h_hpd').flatten()
# hpd = np.append(hpd, metricrw.get_values('distance_h_hpd').flatten())
# hpd = np.append(hpd, metricpr.get_values('distance_h_hpd').flatten())

# out_path = 'data/figs/entropy/singles/bar_diffs.png'
# vis.boxplot_metric_pd('Entropy Diff Annotated over all K\'s and SS', 'Entropy',
#     save_flag=True, save_path=out_path,
#     h_hn=list(hn),
#     h_apd=list(apd),
#     h_hpd=list(hpd)
#     )


# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         out_path = 'data/figs/entropy/lineplots_diff_ann/{}/log{}_k{}.png'.format(short, log, k)
#         vis.line_ploter(steps, 'Entropy Diff Annotated, {}, log={}, k={}'.format(long, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path=out_path, y_lim=(0, 6),
#             h_hn=metric.get_values('distance_h_hn', i, j),
#             h_apd=metric.get_values('distance_h_apd', i, j),
#             h_hpd=metric.get_values('distance_h_hpd', i, j)
#             )


for i, log in enumerate(logs):
    for j, k in enumerate(ks):
        out_path = 'data/figs/entropy/lineplots_diff_true_stripped/{}/log{}_k{}.png'.format(short, log, k)
        vis.line_ploter(steps, 'Entropy Diff Stripped True, {}, log={}, k={}'.format(long, log, k), 'Judgement Points', 'Entropy',
            save_flag=True, save_path=out_path, y_lim=(0, 6),
            h_h=metric.get_values('distance_stripped_h_h', i, j),
            h_hn=metric.get_values('distance_stripped_h_hn', i, j),
            h_apd=metric.get_values('distance_stripped_h_apd', i, j),
            h_hpd=metric.get_values('distance_stripped_h_hpd', i, j)
            )

# for i, log in enumerate(logs):
#     for j, k in enumerate(ks):
#         out_path = 'data/figs/entropy/jsd/{}/log{}_k{}.png'.format(short, log, k)
#         vis.line_ploter(steps, 'JSD, {}, log={}, k={}'.format(long, log, k), 'Judgement Points', 'Entropy',
#             save_flag=True, save_path=out_path,
#             jsd=metric.get_values('jensen_shannon_divergence', i, j),
#             jsd_h=metric.get_values('jsd_approximation_entropy', i, j),
#             jsd_kld=metric.get_values('jsd_approximation_kld', i, j)
#             )
