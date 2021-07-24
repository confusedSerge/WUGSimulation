import os
import pickle
import numpy as np

from csv import writer
from graphs.base_graph import BaseGraph
from analysis.metric_runner import MetricRunner
from analysis.metric_results import MetricResults
import analysis.metrics as am
import analysis.comparison_metrics as acm
import analysis.utils.utils as util
from simulation.clustering.clustering_strategy import new_correlation_clustering

# get all graphs paths
path_true = 'data/graphs/kw29/bigdata/true'
paths_to_true = []
for _, _, files in os.walk(path_true):
    file : str
    for file in files:
        if file.endswith('.graph'):
            paths_to_true.append((file, '{}/{}'.format(path_true, file)))
paths_to_true.sort()

# calc and csv
header = ['name', 'entropy', 'entropy_normalized']
file_path_csv = 'data/graphs/kw29/bigdata/metric/base.csv'
with open(file_path_csv, 'w+', newline='') as file:
    csv_writer = writer(file)
    csv_writer.writerow(header)

    for i, (name, path) in enumerate(paths_to_true):
        print("Graph {}: name {}".format(i + 1, name))

        with open(path, 'rb') as file:
            graph : BaseGraph = pickle.load(file)
        file.close()

        csv_writer.writerow([name.replace('.graph', ''), am.entropy_clustered(graph, {}), am.entropy_clustered_normalized(graph, {})])

file.close()