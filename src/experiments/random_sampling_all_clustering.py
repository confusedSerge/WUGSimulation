import os
import sys
import pickle
import numpy as np

sys.path.append('src')

from graphs.base_graph import BaseGraph
from graphs.annotated_graph import AnnotatedGraph

from simulation.simulation import Simulation

from simulation.sampling.sampling import Sampling
from simulation.sampling.sampling_strategy import page_rank
from simulation.sampling.annotator import Annotator

from simulation.clustering.clustering import Clustering
from simulation.clustering.clustering_strategy import new_correlation_clustering, connected_components_clustering, chinese_whisper_clustering, louvain_method_clustering, sbm_clustering

from simulation.utils.metric_listener import MetricListener
from analysis.comparison_metrics import adjusted_rand_index, jensen_shannon_divergence
from analysis.metrics import cluster_number, bootstraping_jsd, bootstraping_perturbation_ari

from simulation.utils.intermediate_save_listener import IntermediateSaveListener


def randomsampling_sim(graph_path: str, rounds: int, annotations_per_edge: int):
    path_true = 'experiment_data/{}'.format(graph_path)
    path_out = '{}_results/rs/{}'.format(path_true, '{}')
    file_suffix = '.graph'

    # get all graphs paths
    paths_to_true = []
    for _, _, files in os.walk(path_true):
        file: str
        for file in files:
            if file.endswith(file_suffix):
                paths_to_true.append((file, '{}/{}'.format(path_true, file)))
    paths_to_true.sort()

    # Run Sim
    checkpoints = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

    name: str
    path: str
    for i, (name, path) in enumerate(paths_to_true):
        for _round in range(rounds):
            print("Graph {}: Round: {}".format(name, _round + 1))
            with open(path, 'rb') as file:
                graph: BaseGraph = pickle.load(file)
            file.close()

            annotated_graph = AnnotatedGraph(graph.get_number_nodes())

            # Sampler/s
            sampling_step = Sampling(annotations_per_edge=annotations_per_edge).add_sampling_strategie(page_rank, {'sample_size': int(10 / annotations_per_edge), 'start': annotated_graph.get_last_added_node, 'tp_coef': 1.0})\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .set_annotator_dist('random')

            bootstraping_jsd_param = {'sample_size': 100}

            # Listeners CC Split
            clustering_step_cc_split = Clustering().add_clustering_strategy(new_correlation_clustering, {'weights': 'edge_soft_weight', 'max_attempts': 1000, 'max_iters': 5000, 'split_flag': True, 'cores': 0})

            name_metric_rs = '{}-{}-{}-randomsampling_cc_split'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-randomsampling_cc_split_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_cc_split = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cc_split)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': new_correlation_clustering, 'clustering_params': {'weights': 'edge_soft_weight', 'max_attempts': 1000, 'max_iters': 5000, 'split_flag': True, 'cores': 0}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_cc_split = IntermediateSaveListener()\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cc_split)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                .save_draw()

            # Listeners CC No-Split
            clustering_step_cc_nosplit = Clustering().add_clustering_strategy(new_correlation_clustering, {'weights': 'edge_soft_weight', 'max_attempts': 1000, 'max_iters': 5000, 'split_flag': False, 'cores': 0})

            name_metric_rs = '{}-{}-{}-randomsampling_cc_nosplit'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-randomsampling_cc_nosplit_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_cc_nosplit = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cc_nosplit)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': new_correlation_clustering, 'clustering_params': {'weights': 'edge_soft_weight', 'max_attempts': 1000, 'max_iters': 5000, 'split_flag': False, 'cores': 0}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_cc_nosplit = IntermediateSaveListener()\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cc_nosplit)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                .save_draw()

            # Listeners CCC
            clustering_step_ccc = Clustering().add_clustering_strategy(connected_components_clustering, {'weights': 'edge_soft_weight'})

            name_metric_rs = '{}-{}-{}-randomsampling_ccc'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-randomsampling_ccc_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_ccc = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_ccc)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': connected_components_clustering, 'clustering_params': {'weights': 'edge_soft_weight'}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_ccc = IntermediateSaveListener()\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_ccc)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                .save_draw()

            # Listeners CW
            clustering_step_cw = Clustering().add_clustering_strategy(chinese_whisper_clustering, {'weights': 'edge_soft_weight'})

            name_metric_rs = '{}-{}-{}-randomsampling_cw'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-randomsampling_cw_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_cw = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cw)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': chinese_whisper_clustering, 'clustering_params': {'weights': 'edge_soft_weight'}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_cw = IntermediateSaveListener()\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cw)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                .save_draw()

            # Listeners LM
            clustering_step_lm = Clustering().add_clustering_strategy(louvain_method_clustering, {'fmap': lambda x: x - 2.5 if x > 2.5 else 0})

            name_metric_rs = '{}-{}-{}-randomsampling_lm'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-randomsampling_lm_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_lm = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_lm)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': louvain_method_clustering, 'clustering_params': {'fmap': lambda x: x - 2.5 if x > 2.5 else 0}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_lm = IntermediateSaveListener()\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_lm)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                .save_draw()

            # Listeners SBM
            clustering_step_sbm = Clustering().add_clustering_strategy(sbm_clustering, {})

            name_metric_rs = '{}-{}-{}-randomsampling_sbm'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-randomsampling_sbm_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_sbm = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_sbm)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': sbm_clustering, 'clustering_params': {}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_sbm = IntermediateSaveListener()\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_sbm)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                .save_draw()

            # Simulation
            simulation = Simulation(600, break_on_sc=False, verbose=True)\
                .add_step(sampling_step)\
                .add_step(metric_cc_split)\
                .add_step(listener_cc_split)\
                .add_step(metric_cc_nosplit)\
                .add_step(listener_cc_nosplit)\
                .add_step(metric_ccc)\
                .add_step(listener_ccc)\
                .add_step(metric_cw)\
                .add_step(listener_cw)\
                .add_step(metric_lm)\
                .add_step(listener_lm)\
                .add_step(metric_sbm)\
                .add_step(listener_sbm)\

            simulation.run(graph, annotated_graph)


if __name__ == '__main__':
    randomsampling_sim(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
