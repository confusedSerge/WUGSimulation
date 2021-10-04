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
from simulation.sampling.adv_sampling_strategy import dwug_sampling
from simulation.sampling.annotator import Annotator

from simulation.clustering.clustering import Clustering
from simulation.clustering.clustering_strategy import new_correlation_clustering, connected_components_clustering, chinese_whisper_clustering, louvain_method_clustering, sbm_clustering

from simulation.utils.metric_listener import MetricListener
from analysis.comparison_metrics import adjusted_rand_index, jensen_shannon_divergence
from analysis.metrics import cluster_number, bootstraping_jsd, bootstraping_perturbation_ari

from simulation.utils.intermediate_save_listener import IntermediateSaveListener


def dwug_sim(graph_path: str, rounds: int, annotations_per_edge: int):
    error = 0
    path_true = 'experiment_data/{}'.format(graph_path)
    path_out = '{}_results/dwug/{}'.format(path_true, '{}')
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
            sampling_step = Sampling(annotations_per_edge=annotations_per_edge).add_adv_sampling_strategie(dwug_sampling, {'percentage_nodes': int(10 / annotations_per_edge), 'percentage_edges': int(10 / annotations_per_edge), 'min_size_mc': 2, 'num_flag': True}, None)\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30))\
                .set_annotator_dist('random')

            bootstraping_jsd_param = {'sample_size': 100}

            # Listeners CCC
            clustering_step_ccc = Clustering().add_clustering_strategy(connected_components_clustering, {'weights': 'edge_soft_weight'})

            name_metric_rs = '{}-{}-{}-dwug_ccc'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            name_metric_rs_judgement = '{}-{}-{}-dwug_ccc_j'.format(name.replace(file_suffix, ''), _round + 1, annotations_per_edge)
            
            if os.path.exists(os.path.join(os.getcwd(), path_out.format(name_metric_rs))):
                print('Graph Exists, Skipping')
                continue

            metric_ccc = MetricListener(name_metric_rs, path_out.format(name_metric_rs), checkpoints, annotated_graph.get_num_added_edges, tail_write=True)\
                .skip_only_zeros()\
                .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
                .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': connected_components_clustering, 'clustering_params': {'weights': 'edge_soft_weight'}})\
                .add_simple_metric('cluster_number', cluster_number, {})\
                .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
                .add_comparison_metric('ari', adjusted_rand_index, {})\

            listener_ccc = IntermediateSaveListener(tail_write=True)\
                .skip_only_zeros()\
                .add_listener(checkpoints, path_out.format(name_metric_rs), name_metric_rs_judgement, annotated_graph.get_num_added_edges)\
                .save_draw()

            # Simulation
            simulation = Simulation(600, break_on_sc=True, tail_write=True, verbose=True)\
                .add_step(sampling_step)\
                .add_step(clustering_step_ccc)\
                .add_step(metric_ccc)\
                .add_step(listener_ccc)\


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
    dwug_sim(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
