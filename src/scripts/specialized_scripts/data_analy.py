import pickle
import numpy as np

from analysis.metric_results import MetricResults

from visualization.metric_vis import heatmap
from visualization.metric_vis import boxplot_metric_pd as boxplot

'''
Important, this is just a sandbox script for generating some plots 
'''

metric_dwug_path = 'data/graphs/sim_graphs/dwug/sim_ks_logsofthard/2021_06_11_16_00/metric/dwug_comb.data'
metric_pr_path = 'data/graphs/sim_graphs/pagerank/sim_ks_logsofthard/2021_06_11_12_45/metric/pagerank_comb.data'
metric_rw_path = 'data/graphs/sim_graphs/randomwalk/sim_ks_logsofthard/2021_06_11_12_47/metric/randomwalk_comb.data'
metric_rs_path = 'data/graphs/sim_graphs/randomsampling/sim_ks_logsofthard/2021_06_11_12_46/metric/randomsampling_comb.data'

# sampling_label = ['dwug', 'pr', 'rw', 'rs']
sampling_label = ['dwug', 'pr', 'rw']
# metrics_ = ['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance']
metrics_ = ['adjusted_randIndex', 'accuracy', 'inverse_jensen_shannon_distance']
k = [1, 3, 5, 10]
c = ['hard', 'soft']

judgments = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

# === Load data===
with open(metric_dwug_path, 'rb') as file:
    metric_dwug : MetricResults = pickle.load(file)
file.close()

with open(metric_pr_path, 'rb') as file:
    metric_pr : MetricResults = pickle.load(file)
file.close()

with open(metric_rw_path, 'rb') as file:
    metric_rw : MetricResults = pickle.load(file)
file.close()

with open(metric_rs_path, 'rb') as file:
    metric_rs : MetricResults = pickle.load(file)
file.close()

# heatmap
# a = []
# for _metric in metrics_:
#     if len(a) == 0:
#         a = metric_dwug.get_values(_metric, 1)
#     else:
#         a = np.append(a, metric_dwug.get_values(_metric, 1), 0)

# for _metric in metrics_:
#         a = np.append(a, metric_pr.get_values(_metric, 1), 0)

# for _metric in metrics_:
#         a = np.append(a, metric_rw.get_values(_metric, 1), 0)

# # for _metric in metrics_:
# #         a = np.append(a, metric_rs.get_values(_metric, 0), 0)

# labels = []
# for s in sampling_label:
#     for m in metrics_:
#         for _k in k:
#             labels.append('{}_{}_k{}'.format(s, m, _k))

# heatmap(np.round(a, 2), 'Heatmap of Sampling + log soft', judgments, labels)

# boxplots
boxplot('Mean Performance over k of DWUG SoftLog', 'Performance', True, 'data/figs/sim_graphs/metric_analysis/loghardsoft_k13510/dwug_bx_kmean_softlog.png',
    adjusted_randIndex=np.round(metric_dwug.mean('adjusted_randIndex', (1), 0), 2),
    purity=np.round(metric_dwug.mean('purity', (1), 0), 2),
    accuracy=np.round(metric_dwug.mean('accuracy', (1), 0), 2),
    inverse_jensen_shannon_distance=np.round(metric_dwug.mean('inverse_jensen_shannon_distance', (1), 0), 2))

boxplot('Mean Performance over k of DWUG HardLog', 'Performance', True, 'data/figs/sim_graphs/metric_analysis/loghardsoft_k13510/dwug_bx_kmean_hardlog.png',
    adjusted_randIndex=np.round(metric_dwug.mean('adjusted_randIndex', (0), 0), 2),
    purity=np.round(metric_dwug.mean('purity', (0), 0), 2),
    accuracy=np.round(metric_dwug.mean('accuracy', (0), 0), 2),
    inverse_jensen_shannon_distance=np.round(metric_dwug.mean('inverse_jensen_shannon_distance', (0), 0), 2))

boxplot('Mean Performance over k of PageRank SoftLog', 'Performance', True, 'data/figs/sim_graphs/metric_analysis/loghardsoft_k13510/pr_bx_kmean_softlog.png',
    adjusted_randIndex=np.round(metric_pr.mean('adjusted_randIndex', (1), 0), 2),
    purity=np.round(metric_pr.mean('purity', (1), 0), 2),
    accuracy=np.round(metric_pr.mean('accuracy', (1), 0), 2),
    inverse_jensen_shannon_distance=np.round(metric_pr.mean('inverse_jensen_shannon_distance', (1), 0), 2))

boxplot('Mean Performance over k of PageRank HardLog', 'Performance', True, 'data/figs/sim_graphs/metric_analysis/loghardsoft_k13510/pr_bx_kmean_hardlog.png',
    adjusted_randIndex=np.round(metric_pr.mean('adjusted_randIndex', (0), 0), 2),
    purity=np.round(metric_pr.mean('purity', (0), 0), 2),
    accuracy=np.round(metric_pr.mean('accuracy', (0), 0), 2),
    inverse_jensen_shannon_distance=np.round(metric_pr.mean('inverse_jensen_shannon_distance', (0), 0), 2))

boxplot('Mean Performance over k of RandomWalk SoftLog', 'Performance', True, 'data/figs/sim_graphs/metric_analysis/loghardsoft_k13510/rw_bx_kmean_softlog.png',
    adjusted_randIndex=np.round(metric_rw.mean('adjusted_randIndex', (1), 0), 2),
    purity=np.round(metric_rw.mean('purity', (1), 0), 2),
    accuracy=np.round(metric_rw.mean('accuracy', (1), 0), 2),
    inverse_jensen_shannon_distance=np.round(metric_rw.mean('inverse_jensen_shannon_distance', (1), 0), 2))

boxplot('Mean Performance over k of RandomWalk HardLog', 'Performance', True, 'data/figs/sim_graphs/metric_analysis/loghardsoft_k13510/rw_bx_kmean_hardlog.png',
    adjusted_randIndex=np.round(metric_rw.mean('adjusted_randIndex', (0), 0), 2),
    purity=np.round(metric_rw.mean('purity', (0), 0), 2),
    accuracy=np.round(metric_rw.mean('accuracy', (0), 0), 2),
    inverse_jensen_shannon_distance=np.round(metric_rw.mean('inverse_jensen_shannon_distance', (0), 0), 2))

boxplot('Mean Performance over k of RandomSampling SoftLog', 'Performance', True, 'data/figs/sim_graphs/metric_analysis/loghardsoft_k13510/rs_bx_kmean_softlog.png',
    adjusted_randIndex=np.round(metric_rs.mean('adjusted_randIndex', (1), 0), 2),
    purity=np.round(metric_rs.mean('purity', (1), 0), 2),
    accuracy=np.round(metric_rs.mean('accuracy', (1), 0), 2),
    inverse_jensen_shannon_distance=np.round(metric_rs.mean('inverse_jensen_shannon_distance', (1), 0), 2))

boxplot('Mean Performance over k of RandomSampling HardLog', 'Performance', True, 'data/figs/sim_graphs/metric_analysis/loghardsoft_k13510/rs_bx_kmean_hardlog.png',
    adjusted_randIndex=np.round(metric_rs.mean('adjusted_randIndex', (0), 0), 2),
    purity=np.round(metric_rs.mean('purity', (0), 0), 2),
    accuracy=np.round(metric_rs.mean('accuracy', (0), 0), 2),
    inverse_jensen_shannon_distance=np.round(metric_rs.mean('inverse_jensen_shannon_distance', (0), 0), 2))