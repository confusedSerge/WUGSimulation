import os
import sys
import pickle
import numpy as np
import csv
import traceback

sys.path.append('src')

from graphs.base_graph import BaseGraph
from graphs.annotated_graph import AnnotatedGraph

from simulation.simulation import Simulation

from simulation.sampling.sampling import Sampling
from simulation.sampling.sampling_strategy import modified_randomwalk
from simulation.sampling.annotator import Annotator

from simulation.clustering.clustering import Clustering
from simulation.clustering.clustering_strategy import new_correlation_clustering, connected_components_clustering, chinese_whisper_clustering, louvain_method_clustering, sbm_clustering

from simulation.utils.metric_listener import MetricListener
from analysis.comparison_metrics import adjusted_rand_index, jensen_shannon_divergence
from analysis.metrics import cluster_number, bootstraping_jsd, bootstraping_perturbation_ari

from simulation.utils.intermediate_save_listener import IntermediateSaveListener


def modifiedrandomwalk_sim(graph_path: str, rounds: int, annotations_per_edge: int):
    error = 0
    path_true = 'experiment_data/{}'.format(graph_path)
    path_out = '{}_results/mrw/{}'.format(path_true, '{}')
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
            sampling_step = Sampling(annotations_per_edge=annotations_per_edge).add_sampling_strategie(modified_randomwalk, {'sample_size': int(10 / annotations_per_edge), 'start': annotated_graph.get_last_added_node, 'conntained_func': annotated_graph.G.nodes})\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .set_annotator_dist('random')

            bootstraping_jsd_param = {'sample_size': 100}

            # Listeners CC Split
            clustering_step_cc_split = Clustering().add_clustering_strategy(new_correlation_clustering, {'weights': 'edge_soft_weight', 'max_attempts': 1000, 'max_iters': 5000, 'split_flag': True, 'cores': 1})

            name_metric_rs = '{}-{}-{}-modifiedrandomwalk_cc_split'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-modifiedrandomwalk_cc_split_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            
            if os.path.exists(os.path.join(os.getcwd(), path_out.format(name_metric_rs))):
                print('Graph Exists, Skipping')
                continue

            metric_cc_split = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges, tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cc_split)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': new_correlation_clustering, 'clustering_params': {'weights': 'edge_soft_weight', 'max_attempts': 1000, 'max_iters': 5000, 'split_flag': True, 'cores': 1}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_cc_split = IntermediateSaveListener(tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cc_split)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                

            # Listeners CC No-Split
            clustering_step_cc_nosplit = Clustering().add_clustering_strategy(new_correlation_clustering, {'weights': 'edge_soft_weight', 'max_attempts': 1000, 'max_iters': 5000, 'split_flag': False, 'cores': 1})

            name_metric_rs = '{}-{}-{}-modifiedrandomwalk_cc_nosplit'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-modifiedrandomwalk_cc_nosplit_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_cc_nosplit = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges, tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cc_nosplit)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': new_correlation_clustering, 'clustering_params': {'weights': 'edge_soft_weight', 'max_attempts': 1000, 'max_iters': 5000, 'split_flag': False, 'cores': 1}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_cc_nosplit = IntermediateSaveListener(tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cc_nosplit)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                

            # Listeners CCC
            clustering_step_ccc = Clustering().add_clustering_strategy(connected_components_clustering, {'weights': 'edge_soft_weight'})

            name_metric_rs = '{}-{}-{}-modifiedrandomwalk_ccc'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-modifiedrandomwalk_ccc_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_ccc = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges, tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_ccc)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': connected_components_clustering, 'clustering_params': {'weights': 'edge_soft_weight'}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_ccc = IntermediateSaveListener(tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_ccc)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                

            # Listeners CW
            clustering_step_cw = Clustering().add_clustering_strategy(chinese_whisper_clustering, {'weights': 'edge_soft_weight'})

            name_metric_rs = '{}-{}-{}-modifiedrandomwalk_cw'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-modifiedrandomwalk_cw_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_cw = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges, tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cw)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': chinese_whisper_clustering, 'clustering_params': {'weights': 'edge_soft_weight'}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_cw = IntermediateSaveListener(tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_cw)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                

            # Listeners LM
            clustering_step_lm = Clustering().add_clustering_strategy(louvain_method_clustering, {})

            name_metric_rs = '{}-{}-{}-modifiedrandomwalk_lm'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-modifiedrandomwalk_lm_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_lm = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges, tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_lm)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': louvain_method_clustering, 'clustering_params': {}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_lm = IntermediateSaveListener(tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_lm)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                

            # Listeners LM
            clustering_step_sbm = Clustering().add_clustering_strategy(sbm_clustering, {})

            name_metric_rs = '{}-{}-{}-modifiedrandomwalk_sbm'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-modifiedrandomwalk_sbm_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)

            metric_sbm = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges, tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_sbm)\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': sbm_clustering, 'clustering_params': {}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_sbm = IntermediateSaveListener(tail_write=True)\
                .skip_only_zeros()\
                .add_preprocessing_step(clustering_step_sbm)\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                

            # Simulation
            simulation = Simulation(600, break_on_sc=True, tail_write=True, verbose=True)\
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


            try:
                simulation.run(graph, annotated_graph)
            except Exception as ex:
                _out_error = path_out.format('graph_error.csv')

                if error == 0:
                    header = ['Name', 'Round', 'Error']
                    with open(_out_error, 'w+', newline='') as error_file:
                        writer = csv.writer(error_file)
                        writer.writerow(header)
                        error += 1
                    error_file.close()

                with open(_out_error, 'a+', newline='') as error_file:
                    writer = csv.writer(error_file)
                    writer.writerow([str(name), str(_round + 1), traceback.format_exception(type(ex), ex, ex.__traceback__)])
                error_file.close()


if __name__ == '__main__':
    modifiedrandomwalk_sim(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
