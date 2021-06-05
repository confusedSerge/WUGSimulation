import pickle

from analysis.metric_results import MetricResults
from visualization.metric_vis import bar_metric_pd
from visualization.metric_vis import boxplot_metric_pd


"""
This is a special script for running some sims.
"""

metric_data_path = 'data/graphs/sim_graphs/randomwalk/2021_06_05_10_50/metric/RandomWalk.data'

with open(metric_data_path, 'rb') as file:
    _results: MetricResults = pickle.load(file)
file.close()

_metrics = {}
for k in ['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance']:
    _metrics[k] = _results.get_values(k, 3)

judgements_anpoint = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
# bar_metric_pd(judgements_anpoint, 'Performance of RW k=10', 'Performance', 
#     adjusted_randIndex=_metrics['adjusted_randIndex'], purity=_metrics['purity'], accuracy=_metrics['accuracy'], inverse_jensen_shannon_distance=_metrics['inverse_jensen_shannon_distance'])

_metrics = {}
for k in ['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance']:
    _metrics[k] = _results.get_values(k, None, 4)

boxplot_metric_pd(['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance'], 'Performance RandomWalk at judgementspoint = 50', 
    'Performance', adjusted_randIndex=_metrics['adjusted_randIndex'], purity=_metrics['purity'], accuracy=_metrics['accuracy'], inverse_jensen_shannon_distance=_metrics['inverse_jensen_shannon_distance'])

print('done')