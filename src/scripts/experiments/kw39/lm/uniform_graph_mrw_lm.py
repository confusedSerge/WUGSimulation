import numpy as np
from scipy.stats import lognorm

from graphs.annotated_graph import AnnotatedGraph
from graphs.simulation_graph import SimulationGraph
from graphs.utils.distribution import Binomial

from simulation.simulation import Simulation

from simulation.sampling.sampling import Sampling
from simulation.sampling.sampling_strategy import modified_randomwalk
from simulation.sampling.annotator import Annotator

from simulation.clustering.clustering import Clustering
from simulation.clustering.clustering_strategy import louvain_method_clustering

from simulation.utils.metric_listener import MetricListener
from analysis.comparison_metrics import adjusted_rand_index, jensen_shannon_divergence
from analysis.metrics import cluster_number, bootstraping_jsd, bootstraping_perturbation_ari

from simulation.utils.intermediate_save_listener import IntermediateSaveListener

path_out = 'data/graphs/kw39/sim/log09_mrw_shifted_cut/{}'
file_suffix = '.graph'
rounds = 1


def _log_dispensation_communities(num_nodes: int, num_communities: int, params: dict) -> list:
    """
    Uses a lognorm probability density to calculate the sizes of each community

    Args:
        :params num_nodes: number of nodes to dispensate
        :params num_communities: number of communities (buckets) to dispensate to
        :params std_dev: standard deviation of the log_norm distribution
        :return list: dispensated nodes
    """
    assert type(params) == dict
    std_dev = params.get('std_dev', 0.5)

    community_split = lognorm.pdf(np.linspace(1, num_communities, num_communities), std_dev) * num_nodes
    community_split = [int(x) if int(x) > 0 else 1 for x in community_split]

    community_split[0] += num_nodes - sum(community_split)

    return community_split


def fmap_func(x):
    # unary
    # return 1

    # binary
    # return round((x - 1) / 3)

    # normalized
    # return (x - 1) / 3

    # shifted_cut
    y = x - 2.5
    return y if y > 0 else 0


# Run Sim
checkpoints = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
uniform_sizes = [_log_dispensation_communities(100, i, {'std_dev': 0.9}) for i in range(1, 11)]

name: str
path: str
for i, name in enumerate(['log09_clusters_n100_k{}'.format(i + 1) for i in range(len(uniform_sizes))]):
    # if i < 9:
    #     continue

    for _round in range(rounds):
        print("Graph {}: Round: {}".format(name, _round))

        graph = SimulationGraph(uniform_sizes[i], None, Binomial(3, 1, len(uniform_sizes[i])))
        annotated_graph = AnnotatedGraph(graph.get_number_nodes())

        # Sampler/s
        sampling_step = Sampling(annotations_per_edge=1).add_sampling_strategie(modified_randomwalk, {'sample_size': 10, 'start': annotated_graph.get_last_added_node, 'conntained_func': annotated_graph.G.nodes})\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5))\
            .set_annotator_dist('random')

        bootstraping_jsd_param = {'sample_size': 100}

        # Listeners LM
        clustering_step_lm = Clustering().add_clustering_strategy(louvain_method_clustering, {'fmap': fmap_func})

        metric_lm = MetricListener('{}-{}-modifiedrandomwalk_1_lm'.format(name.replace(file_suffix, ''), _round), path_out.format(
            '{}-{}-modifiedrandomwalk_1_lm'.format(name.replace(file_suffix, ''), _round)), checkpoints, annotated_graph.get_num_added_edges)\
            .add_preprocessing_step(clustering_step_lm)\
            .add_simple_metric('bootstrap_jsd', bootstraping_jsd, bootstraping_jsd_param)\
            .add_simple_metric('gambette_01', bootstraping_perturbation_ari, {'share': 0.1, 'clustering_func': louvain_method_clustering, 'clustering_params': {'fmap': fmap_func}})\
            .add_simple_metric('cluster_number', cluster_number, {})\
            .add_comparison_metric('jsd', jensen_shannon_divergence, {})\
            .add_comparison_metric('ari', adjusted_rand_index, {})\

        listener_lm = IntermediateSaveListener()\
            .add_listener(checkpoints, path_out.format('{}-{}-modifiedrandomwalk_1_lm'.format(name.replace(file_suffix, ''), _round)),
                          '{}-{}-modifiedrandomwalk_1_lm_j'.format(name.replace(file_suffix, ''), _round), annotated_graph.get_num_added_edges)\
            .save_draw()\
            .add_preprocessing_step(clustering_step_lm)

        simulation = Simulation(600, break_on_sc=False, verbose=True)\
            .add_step(sampling_step)\
            .add_step(metric_lm)\
            .add_step(listener_lm)

        simulation.run(graph, annotated_graph)
