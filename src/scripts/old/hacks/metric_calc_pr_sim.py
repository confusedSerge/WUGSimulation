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

def get_metric_runner():
    metric_runner = MetricRunner(name='Runner', info='None')
    # entropy metrics
    metric_runner.add_simple_metric(am.entropy_clustered, {})\
        .add_comparison_metric(acm.stripped_entropy, {}, 'stripped_entropy_clustered')\
        .add_simple_metric(am.entropy_approximation, {})\
        .add_simple_metric(am.apd, {'sample_size': 500})\
        .add_simple_metric(am.hpd, {'sample_size': 500})

    # entropy normalized
    metric_runner.add_simple_metric(am.entropy_clustered_normalized, {})\
        .add_comparison_metric(acm.stripped_entropy_normalized, {}, 'stripped_entropy_clustered_normalized')\
        .add_simple_metric(am.entropy_approximation_normalized, {})\
        .add_simple_metric(am.apd_normalized, {'sample_size': 500})\
        .add_simple_metric(am.hpd_normalized, {'sample_size': 500})

    # jsd
    metric_runner.add_comparison_metric(acm.jensen_shannon_divergence, {})\
        .add_comparison_metric(acm.jsd_approximation_entropy, {})\
        .add_comparison_metric(acm.jsd_approximation_kld, {})\
        .add_comparison_metric(acm.jsd_approximation_apd, {'sample_size': 500})\
        .add_comparison_metric(acm.jsd_approximation_hpd, {'sample_size': 500})
    return metric_runner

def save_to_csv(file_path_out: str, name: str, metric_result: MetricResults):
    with open(file_path_out, 'a+', newline='') as file:
        csv_writer = writer(file)

        for k, v in metric_result.metric_dict.items():
            row = ['{}-{}'.format(name, k)]
            row.extend(v)
            csv_writer.writerow(row)
    file.close()

# create csv out
header = ['name']
judgements = list(range(10, 100, 10))
judgements.extend(list(range(100, 5100, 100)))
header.extend(judgements)

file_path_csv = 'data/graphs/kw29/bigdata/metric/pagerank.csv'
with open(file_path_csv, 'w+', newline='') as file:
    csv_writer = writer(file)
    csv_writer.writerow(header)
file.close()

# find all graphs
sampling_strat = 'pagerank'
path_to_true = 'data/graphs/kw29/bigdata/true/{}.graph'
path_all_pagerank = 'data/graphs/kw29/bigdata/sim/pagerank/'

true_ann = []
directory : str
for directory, sub_directories, files in os.walk(path_all_pagerank):
    if len(sub_directories) == 0:
        true_path = path_to_true.format(directory.split('/')[-1])
        true_ann.append((directory.split('/')[-1].replace('.graph', ''), true_path, directory))
true_ann.sort()

for name_true, true_path, annotated_graph_path in true_ann:
    print('Currently working on: {}, {}, {}'.format(name_true, true_path, annotated_graph_path))
    # Load graphs to use
    annotated_graphs = []
    sort_func = lambda x: x.get_num_added_edges()
    for _, _, files in os.walk(annotated_graph_path):
        for file in files:
            if file.endswith('.graph'):
                with open('{}/{}'.format(annotated_graph_path, file), 'rb') as file_graph:
                    annotated_graphs.append(pickle.load(file_graph))
                file_graph.close()
    annotated_graphs.sort(key=sort_func)

    with open(true_path, 'rb') as file_graph:
        true_graphs = [pickle.load(file_graph)]*len(annotated_graphs)
    file_graph.close()
    print('Loaded Graphs')
    
    # Name tag in csv
    name = '{}-{}'.format(name_true, sampling_strat)

    # clustering for simulated graphs, so that metrics can be calculated, with islands
    for i in range(len(annotated_graphs)):
        _cl = new_correlation_clustering(annotated_graphs[i], {'weights': 'edge_soft_weight', 'max_attempts':50, 'max_iters':100, 'split_flag': True})
        annotated_graphs[i].update_community_nodes_membership(_cl)
    print('Clustered Graphs')

    metric_runner = get_metric_runner()
    metric_runner.run(annotated_graphs, true_graphs)
    print('Calculated Metrics')

    save_to_csv(file_path_csv, '{}-{}'.format(name, 'split'), metric_runner.metric_result)

    # clustering for simulated graphs, so that metrics can be calculated, without islands
    for i in range(len(annotated_graphs)):
        _cl = new_correlation_clustering(annotated_graphs[i], {'weights': 'edge_soft_weight', 'max_attempts':50, 'max_iters':100, 'split_flag': False})
        annotated_graphs[i].update_community_nodes_membership(_cl)
    print('Clustered Graphs')

    metric_runner = get_metric_runner()
    metric_runner.run(annotated_graphs, true_graphs)
    print('Calculated Metrics')

    save_to_csv(file_path_csv, '{}-{}'.format(name, 'no_split'), metric_runner.metric_result)

