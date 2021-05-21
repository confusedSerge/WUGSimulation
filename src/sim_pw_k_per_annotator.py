import pickle
import numpy as np

from visualization.graph_visualization import draw_graph_graphviz

from graphs.wu_graph_sampler import WUGraphSampler
from graphs.wu_annotator_graph import WUAnnotatorGraph
from graphs.wu_annotator_simulation_graph import WUAnnotatorSimulationGraph
from graphs.utils.annotator import Annotator

from simulation.simulation import *
from simulation.sampling_strategy import random_sampling
from simulation.sampling_strategy import page_rank_per_annotator
from simulation.clustering_strategy import correlation_clustering
from simulation.stopping_criterion import number_edges_found

from analysis.analyzer import analyze
from analysis.metrics import *

from visualization.metric_vis import bar_metric
from visualization.metric_vis import threed_line_ploter

title_0 = 'vs'
title_pr = 'pageRank'

nodes = 100
com = [*range(1, 11)]
num_ann = 5


sampler = WUGraphSampler(nodes, com, ('log', {'std_dev': 0.99}), ['binomial', 3, 0.99])
for i in range(num_ann):
    sampler.add_annotator(Annotator(np.random.poisson, 0.2, 1))
true_wug_gen = sampler.sample_wug_annotator_generator()

metrics_dict = {'adjusted_randIndex': {}, 'purity': {}, 'accuracy': {}, 'inverse_jensen_shannon_distance': {}}

edges = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]

for i, true_wug in enumerate(true_wug_gen):
    draw_graph_graphviz(true_wug, save_flag=True, path='data/figs/other/true_graph_{}_n{}_k{}.png'.format(title_0, nodes, i + 1))
    simulation_wug_pr = WUAnnotatorSimulationGraph(true_wug.get_number_nodes(), num_ann)

    for edge_found in edges:

        # ===PR Sim===
        print('started sim pr')
        true_wug, simulation_wug_pr, max_iter_flag = simulation(true_wug, simulation_wug_pr, max_iter=500,
                            sampling_strategy=page_rank_per_annotator, sampling_params={'simGraph': simulation_wug_pr, 'sample_size': 2, 'tp_coef': 0.1},
                            #  clustering_strategy=correlation_clustering, clustering_params={},
                            stopping_criterion=number_edges_found,  stopping_params={'number_edges': edge_found})
        print('ended sim pr')

        print('clustering pr')
        simulation_wug_pr.update_community_nodes_membership(correlation_clustering(simulation_wug_pr, {'weights': 'edge_soft_weight'}))
        print('clustering done pr')

        draw_graph_graphviz(simulation_wug_pr, save_flag=True, path='data/figs/other/sim_graph_{}_edges_{}_n{}_k{}.png'.format(title_pr, edge_found, nodes, i + 1))

        tmp = analyze(true_wug, simulation_wug_pr, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), inverse_jensen_shannon_distance=(inverse_jensen_shannon_distance, {}))

        for k, v in tmp.items():
            if metrics_dict[k].get(i + 1, None) == None:
                metrics_dict[k][i + 1] = []
            metrics_dict[k][i + 1].append(v)

with open('data/metric_pr.data', 'wb') as file:
    pickle.dump(metrics_dict, file)
file.close()

# for edge in edges:
#     bar_metric(['PageRank', 'RandomWalk', 'RandomSampling'], 'Bar Graph of sim', 'Performance', True, 'data/figs/other/bar_plot_{}_n{}_k{}.png'.format(edge, nodes, com),
#         adj_randIndex=metrics_dict[edge]['adjusted_randIndex'], purity=metrics_dict[edge]['purity'], accuracy=metrics_dict[edge]['accuracy'], jensenshannon=metrics_dict[edge]['inverse_jensen_shannon_distance'])

threed_line_ploter(edges, '#Edges', com, '#Communities', 'Performance', 'Simulation PageRank', 
    adj_randIndex=metrics_dict['adjusted_randIndex'], purity=metrics_dict['purity'], accuracy=metrics_dict['accuracy'], jensenshannon=metrics_dict['inverse_jensen_shannon_distance'])