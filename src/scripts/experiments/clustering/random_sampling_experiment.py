import os
import pickle
import numpy as np

from graphs.base_graph import BaseGraph
from graphs.annotated_graph import AnnotatedGraph

from simulation.simulation import Simulation

from simulation.sampling.sampling import Sampling
from simulation.sampling.sampling_strategy import page_rank
from simulation.sampling.annotator import Annotator

from simulation.clustering.clustering import Clustering
from simulation.clustering.clustering_strategy import new_correlation_clustering, connected_components_clustering, chinese_whisper_clustering, louvain_method_clustering

from simulation.utils.metric_listener import MetricListener
from analysis.comparison_metrics import adjusted_rand_index, jensen_shannon_divergence
from analysis.metrics import cluster_number, bootstraping_jsd, bootstraping_perturbation_ari

from simulation.utils.intermediate_save_listener import IntermediateSaveListener

path_true = 'data/graphs/kw32/simulation_graphs'
path_out = 'data/graphs/kw32/sim/randomsampling/{}'
file_suffix = '.graph'
rounds = 10

# get all graphs paths
paths_to_true = []
for _, _, files in os.walk(path_true):
    file: str
    for file in files:
        if file.endswith(file_suffix):
            paths_to_true.append((file, '{}/{}'.format(path_true, file)))
paths_to_true.sort()

new_path = []
for name, path in paths_to_true:
    if name.startswith('n100_k3_log0.9') or name.startswith('n100_k7'):
        new_path.append((name, path))
paths_to_true = new_path
for name, path in paths_to_true:
    print(name, path)

# Run Sim
checkpoints = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

name: str
path: str
for i, (name, path) in enumerate(paths_to_true):
    for _round in range(rounds):
        print("Graph {}: Round: {}".format(name, _round))
        with open(path, 'rb') as file:
            graph: BaseGraph = pickle.load(file)
        file.close()

        annotated_graph = AnnotatedGraph(graph.get_number_nodes())

        # Sampler/s
        sampling_step = Sampling().add_sampling_strategie(page_rank, {'sample_size': 10, 'start': annotated_graph.get_last_added_node, 'tp_coef': 1.0})\
            .add_annotator(Annotator(np.random.poisson, [0.35], 1, 4, 0.5))\
            .add_annotator(Annotator(np.random.poisson, [0.35], 1, 4, 0.5))\
            .add_annotator(Annotator(np.random.poisson, [0.35], 1, 4, 0.5))\
            .add_annotator(Annotator(np.random.poisson, [0.35], 1, 4, 0.5))\
            .add_annotator(Annotator(np.random.poisson, [0.35], 1, 4, 0.5))\
            .set_annotator_dist('random')

        bootstraping_jsd_param = {'sample_size': 50}

        # Listeners CC Split
        clustering_step_cc_split = Clustering().add_clustering_strategy(new_correlation_clustering, {
            'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': np.inf, 'split_flag': True})

        metric_cc_split = MetricListener('{}-{}-randomsampling_cc_split'.format(name.replace(file_suffix, ''), _round), path_out.format(
            '{}-{}-randomsampling_cc_split'.format(name.replace(file_suffix, ''), _round)), checkpoints, annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_cc_split)\
            .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
            .add_simple_metric('gambette', bootstraping_perturbation_ari, {'share': 1.0, 'clustering_func': new_correlation_clustering, 'clustering_params': {
                'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': True}})\
            .add_simple_metric('cluster_number', cluster_number, {})\
            .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
            .add_comparison_metric('ari', adjusted_rand_index, {})\

        listener_cc_split = IntermediateSaveListener()\
            .add_listener(checkpoints, path_out.format('{}-{}-randomsampling_cc_split'.format(name.replace(file_suffix, ''), _round)),
                          '{}-{}-randomsampling_cc_split_j'.format(name.replace(file_suffix, ''), _round), annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_cc_split)

        # Listeners No Split
        clustering_step_cc_nsplit = Clustering().add_clustering_strategy(new_correlation_clustering, {
            'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': np.inf, 'split_flag': False})

        metric_cc_nsplit = MetricListener('{}-{}-randomsampling_cc_nosplit'.format(name.replace(file_suffix, ''), _round), path_out.format(
            '{}-{}-randomsampling_cc_nosplit'.format(name.replace(file_suffix, ''), _round)), checkpoints, annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_cc_nsplit)\
            .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
            .add_simple_metric('gambette', bootstraping_perturbation_ari, {'share': 1.0, 'clustering_func': new_correlation_clustering, 'clustering_params': {
                'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': False}})\
            .add_simple_metric('cluster_number', cluster_number, {})\
            .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
            .add_comparison_metric('ari', adjusted_rand_index, {})\

        listener_cc_nsplit = IntermediateSaveListener()\
            .add_listener(checkpoints, path_out.format('{}-{}-randomsampling_cc_nosplit'.format(name.replace(file_suffix, ''), _round)),
                          '{}-{}-randomsampling_cc_nosplit_j'.format(name.replace(file_suffix, ''), _round), annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_cc_nsplit)

        # Listeners CC
        clustering_step_ccc = Clustering().add_clustering_strategy(connected_components_clustering, {'weights': 'edge_soft_weight'})

        metric_ccc = MetricListener('{}-{}-randomsampling_ccc'.format(name.replace(file_suffix, ''), _round), path_out.format(
            '{}-{}-randomsampling_ccc'.format(name.replace(file_suffix, ''), _round)), checkpoints, annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_ccc)\
            .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
            .add_simple_metric('gambette', bootstraping_perturbation_ari, {'share': 1.0, 'clustering_func': connected_components_clustering, 'clustering_params': {'weights': 'edge_soft_weight'}})\
            .add_simple_metric('cluster_number', cluster_number, {})\
            .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
            .add_comparison_metric('ari', adjusted_rand_index, {})\

        listener_ccc = IntermediateSaveListener()\
            .add_listener(checkpoints, path_out.format('{}-{}-randomsampling_ccc'.format(name.replace(file_suffix, ''), _round)),
                          '{}-{}-randomsampling_ccc_j'.format(name.replace(file_suffix, ''), _round), annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_ccc)

        # Listeners CW
        clustering_step_cw = Clustering().add_clustering_strategy(chinese_whisper_clustering, {'weights': 'edge_soft_weight'})

        metric_cw = MetricListener('{}-{}-randomsampling_cw'.format(name.replace(file_suffix, ''), _round), path_out.format(
            '{}-{}-randomsampling_cw'.format(name.replace(file_suffix, ''), _round)), checkpoints, annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_cw)\
            .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
            .add_simple_metric('gambette', bootstraping_perturbation_ari, {'share': 1.0, 'clustering_func': chinese_whisper_clustering, 'clustering_params': {'weights': 'edge_soft_weight'}})\
            .add_simple_metric('cluster_number', cluster_number, {})\
            .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
            .add_comparison_metric('ari', adjusted_rand_index, {})\

        listener_cw = IntermediateSaveListener()\
            .add_listener(checkpoints, path_out.format('{}-{}-randomsampling_cw'.format(name.replace(file_suffix, ''), _round)),
                          '{}-{}-randomsampling_cw_j'.format(name.replace(file_suffix, ''), _round), annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_cw)

        # Listeners LM
        clustering_step_lm = Clustering().add_clustering_strategy(louvain_method_clustering, {})

        metric_lm = MetricListener('{}-{}-randomsampling_lm'.format(name.replace(file_suffix, ''), _round), path_out.format(
            '{}-{}-randomsampling_lm'.format(name.replace(file_suffix, ''), _round)), checkpoints, annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_lm)\
            .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
            .add_simple_metric('gambette', bootstraping_perturbation_ari, {'share': 1.0, 'clustering_func': louvain_method_clustering, 'clustering_params': {}})\
            .add_simple_metric('cluster_number', cluster_number, {})\
            .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
            .add_comparison_metric('ari', adjusted_rand_index, {})\

        listener_lm = IntermediateSaveListener()\
            .add_listener(checkpoints, path_out.format('{}-{}-randomsampling_lm'.format(name.replace(file_suffix, ''), _round)),
                          '{}-{}-randomsampling_lm_j'.format(name.replace(file_suffix, ''), _round), annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_lm)

        simulation = Simulation(600, break_on_sc=False, verbose=True)\
            .add_step(sampling_step)\
            .add_step(metric_cc_split)\
            .add_step(listener_cc_split)\
            .add_step(metric_cc_nsplit)\
            .add_step(listener_cc_nsplit)\
            .add_step(metric_ccc)\
            .add_step(listener_ccc)\
            .add_step(metric_cw)\
            .add_step(listener_cw)\
            .add_step(metric_lm)\
            .add_step(listener_lm)

        simulation.run(graph, annotated_graph)
