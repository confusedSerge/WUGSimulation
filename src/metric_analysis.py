import pickle

from analysis.metric_results import MetricResults
from visualization.metric_vis import bar_metric_pd

metric_data_path = 'data/graphs/sim_graphs/randomwalk/2021_06_05_10_18/metric/RandomWalk.data'

with open(metric_data_path, 'rb') as file:
    _results: MetricResults = pickle.load(file)
file.close()

_metrics = {}
for k in ['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance']:
    _metrics[k] = _results.get_values(k, 1)

judgements_anpoint = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
bar_metric_pd(judgements_anpoint, 'Performance of RW k=1', 'Performance', 
    adjusted_randIndex=_metrics['adjusted_randIndex'], purity=_metrics['purity'], accuracy=_metrics['accuracy'], inverse_jensen_shannon_distance=_metrics['inverse_jensen_shannon_distance'])

print('done')