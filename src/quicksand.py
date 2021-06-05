# just a script to play around, test some functionalities, find bugs
import pickle
from graphs.wu_graph import WUGraph
from graphs.wu_simulation_graph import WUSimulationGraph

from simulation.full_simulation import full_simulation
from simulation.sampling.sampling_strategy import dwug_sampling
from simulation.clustering.clustering_strategy import new_correlation_clustering
from simulation.stopping.stopping_criterion import cluster_connected
from simulation.stopping.stopping_criterion import edges_added

from analysis.metric_results import MetricResults
from analysis.analyzer import analyze
from analysis.metrics import *

data = 'data/graphs/true_graphs/2021_06_04_16_08/true_graph_wug_n100_k3_clog_dbinomial.graph'

with open(data, 'rb') as file:
    graph = pickle.load(file)
file.close()

judgements_anpoint = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

simulation_wug_dwug = WUSimulationGraph(graph.get_number_nodes())
simulation_wug_dwug, max_iter_rs, metrics_dwug, metrics_dwug_graphs \
        = full_simulation(trueGraph=graph, simulationGraph=simulation_wug_dwug, max_iter=5000, break_on_sc=False, verbose=True,
                    sampling_strategy=dwug_sampling, sampling_params={'simulationGraph': simulation_wug_dwug, 'percentage_nodes': 0.1, 'percentage_edges': 0.3, 'min_size_mc': 2},
                    clustering_strategy=new_correlation_clustering, clustering_params={'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': False},
                    stopping_criterion=cluster_connected,  stopping_params={'cluster_min_size': 2, 'min_num_edges': 1},
                    analyzing_critertion=edges_added, analyzing_critertion_params=[{'number_edges': x} for x in judgements_anpoint],
                    analyzing_func=analyze, 
                    analyzing_params={'adjusted_randIndex': (adjusted_randIndex, {}), 'purity':(purity, {}), 'accuracy':(accuracy, {}), 'inverse_jensen_shannon_distance':(inverse_jensen_shannon_distance, {})}, 
                    return_graph_flag=True)
