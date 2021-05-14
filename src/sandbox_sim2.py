from visualization.graph_visualization import draw_graph_graphviz

from graphs.wu_graph_sampler import WUGraphSampler
from graphs.wu_graph import WUGraph
from graphs.wu_simulation_graph import WUSimulationGraph

from simulation.simulation import *
from simulation.sampling_strategy import random_sampling
from simulation.clustering_strategy import correlation_clustering
from simulation.stopping_criterion import percentage_edges_found

from analysis.analyzer import analyze
from analysis.metrics import *

from visualization.metric_vis import bar_metric
from visualization.metric_vis import line_ploter

metrics = {}
wug_dists = [[50, 50], [60, 40], [70, 30], [80, 20], [90,10], [95, 5], [99, 1]]

for wug_dist in wug_dists:
    true_wug = WUGraphSampler(100, 2, wug_dist, ['binomial', 3, 0.9]).sample_wug()
    simulation_wug = WUSimulationGraph(true_wug.get_number_nodes())

    print('started sim')
    true_wug, simulation_wug, max_iter_flag = simulation(true_wug, simulation_wug, max_iter=250,
                        sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
                        #  clustering_strategy=correlation_clustering, clustering_params={},
                        stopping_criterion=percentage_edges_found,  stopping_params={'percentage': 0.9, 'number_edges': true_wug.get_number_edges()})
    print('ended sim')

    print('clustering')
    simulation_wug.update_community_nodes_membership(correlation_clustering(simulation_wug, {'weights': 'edge_soft_weight'}))
    print('clustering done')

    draw_graph_graphviz(simulation_wug, save_flag=True, path='data/figs/other/sim_graph_perc_{}_n100_k2_c{}-{}.png'.format(0.9, *wug_dist))
    draw_graph_graphviz(true_wug, save_flag=True, path='data/figs/other/true_graph_perc_{}_n100_k2_c{}-{}.png'.format(0.9, *wug_dist))

    tmp = analyze(true_wug, simulation_wug, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), jensen_shannon_distance=(inverse_jensen_shannon_distance, {}))

    for k, v in tmp.items():
        if metrics.get(k, None) == None:
            metrics[k] = []
        metrics[k].append(v)

bar_metric(wug_dists, 'Bar Graph of sim', 'Dispensation', True, 'data/figs/other/bar_plot_perc_0.9_n100_k2.png',
    adj_randIndex=metrics['adjusted_randIndex'], purity=metrics['purity'], accuracy=metrics['accuracy'], jensenshannon=metrics['inverse_jensen_shannon_distance'])

line_ploter(wug_dists, 'Simulated Graph with Random Sampling', 'Dispensation', 'Performance', True, 'data/figs/other/line_plot_perc_0.9_n100_k2.png', 
    adj_randIndex=metrics['adjusted_randIndex'], purity=metrics['purity'], accuracy=metrics['accuracy'], jensenshannon=metrics['inverse_jensen_shannon_distance'])