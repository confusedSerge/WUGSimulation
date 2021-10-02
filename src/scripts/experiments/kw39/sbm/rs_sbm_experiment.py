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
from simulation.clustering.clustering_strategy import sbm_clustering

from simulation.utils.metric_listener import MetricListener
from analysis.comparison_metrics import adjusted_rand_index, jensen_shannon_divergence
from analysis.metrics import cluster_number, bootstraping_jsd, bootstraping_perturbation_ari

from simulation.utils.intermediate_save_listener import IntermediateSaveListener

path_true = 'data/graphs/kw32/simulation_graphs'
path_out = 'data/graphs/kw39/sim/rs/{}'
file_suffix = '.graph'
rounds = 1

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
        print("Graph {}: Round: {}".format(name, _round))
        with open(path, 'rb') as graph_file:
            graph: BaseGraph = pickle.load(graph_file)
        graph_file.close()

        annotated_graph = AnnotatedGraph(graph.get_number_nodes())

        # Sampler/s
        sampling_step = Sampling(annotations_per_edge=1).add_sampling_strategie(page_rank, {'sample_size': 10, 'start': annotated_graph.get_last_added_node, 'tp_coef': 1.0})\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .set_annotator_dist('random')

        # Listeners sbm_clustering
        clustering_step_sbm = Clustering().add_clustering_strategy(sbm_clustering, {})

        metric_sbm = MetricListener('{}-{}-rs_1_sbm'.format(name.replace(file_suffix, ''), _round), path_out.format(
            '{}-{}-rs_1_sbm'.format(name.replace(file_suffix, ''), _round)), checkpoints, annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_sbm)\
            .add_simple_metric('cluster_number', cluster_number, {})\
            .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
            .add_comparison_metric('ari', adjusted_rand_index, {})\

        listener_sbm = IntermediateSaveListener()\
            .add_listener(checkpoints, path_out.format('{}-{}-rs_1_sbm'.format(name.replace(file_suffix, ''), _round)),
                          '{}-{}-rs_1_sbm_j'.format(name.replace(file_suffix, ''), _round), annotated_graph.get_num_added_edges)\
            .save_draw()\
            .add_preprocessing_step(clustering_step_sbm)

        simulation = Simulation(600, break_on_sc=False, verbose=True)\
            .add_step(sampling_step)\
            .add_step(metric_sbm)\
            .add_step(listener_sbm)\

        simulation.run(graph, annotated_graph)
