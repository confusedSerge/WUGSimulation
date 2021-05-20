from visualization.graph_visualization import draw_graph_graphviz

from graphs.wu_graph_sampler import WUGraphSampler
from graphs.wu_graph import WUGraph
from graphs.wu_simulation_graph import WUSimulationGraph

from simulation.simulation import *
from simulation.sampling_strategy import random_sampling
from simulation.clustering_strategy import correlation_clustering
from simulation.stopping_criterion import number_edges_found

from analysis.analyzer import analyze
from analysis.metrics import *

from visualization.metric_vis import bar_metric
from visualization.metric_vis import line_ploter

title = 'edges'
nodes = 100
com = 3

true_wug = WUGraphSampler(nodes, com, ('log', {'std_dev': 0.99}), ['binomial', 3, 0.9]).sample_wug()
simulation_wug = WUSimulationGraph(true_wug.get_number_nodes())

draw_graph_graphviz(true_wug, save_flag=True, path='data/figs/other/true_graph_{}_n{}_k{}.png'.format(title, nodes, com))
true_wug.save_graph('data/graph_pickle/true_graph_{}_n{}_k{}'.format(title, nodes, com))

metrics = {}
edges = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]

for edge_found in edges:
    print('started sim')
    true_wug, simulation_wug, max_iter_flag = simulation(true_wug, simulation_wug, max_iter=250,
                        sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
                        #  clustering_strategy=correlation_clustering, clustering_params={},
                        stopping_criterion=number_edges_found,  stopping_params={'number_edges': edge_found})
    print('ended sim')

    print('clustering')
    simulation_wug.update_community_nodes_membership(correlation_clustering(simulation_wug, {'weights': 'edge_soft_weight'}))
    print('clustering done')

    draw_graph_graphviz(simulation_wug, save_flag=True, path='data/figs/other/sim_graph_{}_edges_{}_n{}_k{}.png'.format(title, edge_found, nodes, com))
    simulation_wug.save_graph('data/graph_pickle/sim_graph_{}_edges_{}_n{}_k{}'.format(title, edge_found, nodes, com))

    tmp = analyze(true_wug, simulation_wug, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), inverse_jensen_shannon_distance=(inverse_jensen_shannon_distance, {}))

    for k, v in tmp.items():
        if metrics.get(k, None) == None:
            metrics[k] = []
        metrics[k].append(v)

bar_metric(edges, 'Bar Graph of sim', 'Performance', True, 'data/figs/other/bar_plot_{}_n{}_k{}.png'.format(title, nodes, com),
    adj_randIndex=metrics['adjusted_randIndex'], purity=metrics['purity'], accuracy=metrics['accuracy'], jensenshannon=metrics['inverse_jensen_shannon_distance'])

line_ploter(edges, 'Simulated Graph with Random Sampling', 'Percentage', 'Performance', True, 'data/figs/other/line_plot_{}_n{}_k{}.png'.format(title, nodes, com), 
    adj_randIndex=metrics['adjusted_randIndex'], purity=metrics['purity'], accuracy=metrics['accuracy'], jensenshannon=metrics['inverse_jensen_shannon_distance'])