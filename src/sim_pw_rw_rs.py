import pickle

from visualization.graph_visualization import draw_graph_graphviz

from graphs.wu_graph_sampler import WUGraphSampler
from graphs.wu_graph import WUGraph
from graphs.wu_simulation_graph import WUSimulationGraph

from simulation.simulation import *
from simulation.sampling_strategy import random_sampling
from simulation.sampling_strategy import page_rank
from simulation.clustering_strategy import correlation_clustering
from simulation.stopping_criterion import number_edges_found

from analysis.analyzer import analyze
from analysis.metrics import *

from visualization.metric_vis import bar_metric
from visualization.metric_vis import line_ploter

title_0 = 'vs'
title_pr = 'pageRank'
title_rw = 'randomWalk'
title_rs = 'randomSample'
nodes = 100
com = 3

true_wug = WUGraphSampler(nodes, com, ('log', {'std_dev': 0.99}), ['binomial', 3, 0.99]).sample_wug()
simulation_wug_pr = WUSimulationGraph(true_wug.get_number_nodes())
simulation_wug_rw = WUSimulationGraph(true_wug.get_number_nodes())
simulation_wug_rs = WUSimulationGraph(true_wug.get_number_nodes())

draw_graph_graphviz(true_wug, save_flag=True, path='data/figs/other/true_graph_{}_n{}_k{}.png'.format(title_0, nodes, com))

metrics_pr = {}
metrics_rw = {}
metrics_rs = {}

metrics = {}

edges = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]

for edge_found in edges:
    # Prime metric
    metrics[edge_found] = {}

    # ===PR Sim===
    print('started sim pr')
    true_wug, simulation_wug_pr, max_iter_flag = simulation(true_wug, simulation_wug_pr, max_iter=500,
                        sampling_strategy=page_rank, sampling_params={'sample_size': 10, 'start': simulation_wug_pr.get_last_added_node, 'tp_coef': 0.1},
                        #  clustering_strategy=correlation_clustering, clustering_params={},
                        stopping_criterion=number_edges_found,  stopping_params={'number_edges': edge_found})
    print('ended sim pr')

    print('clustering pr')
    simulation_wug_pr.update_community_nodes_membership(correlation_clustering(simulation_wug_pr, {'weights': 'edge_soft_weight'}))
    print('clustering done pr')

    draw_graph_graphviz(simulation_wug_pr, save_flag=True, path='data/figs/other/sim_graph_{}_edges_{}_n{}_k{}.png'.format(title_pr, edge_found, nodes, com))

    tmp = analyze(true_wug, simulation_wug_pr, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), inverse_jensen_shannon_distance=(inverse_jensen_shannon_distance, {}))

    for k, v in tmp.items():
        if metrics_pr.get(k, None) == None:
            metrics_pr[k] = []
        metrics_pr[k].append(v)

        if metrics[edge_found].get(k, None) == None:
            metrics[edge_found][k] = []
        metrics[edge_found][k].append(v)


    # ===RW Sim===
    print('started sim rw')
    true_wug, simulation_wug_rw, max_iter_flag = simulation(true_wug, simulation_wug_rw, max_iter=500,
                        sampling_strategy=page_rank, sampling_params={'sample_size': 10, 'start': simulation_wug_rw.get_last_added_node, 'tp_coef': 0.0},
                        #  clustering_strategy=correlation_clustering, clustering_params={},
                        stopping_criterion=number_edges_found,  stopping_params={'number_edges': edge_found})
    print('ended sim rw')

    print('clustering rw')
    simulation_wug_rw.update_community_nodes_membership(correlation_clustering(simulation_wug_rw, {'weights': 'edge_soft_weight'}))
    print('clustering done rw')

    draw_graph_graphviz(simulation_wug_rw, save_flag=True, path='data/figs/other/sim_graph_{}_edges_{}_n{}_k{}.png'.format(title_rw, edge_found, nodes, com))

    tmp = analyze(true_wug, simulation_wug_rw, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), inverse_jensen_shannon_distance=(inverse_jensen_shannon_distance, {}))

    for k, v in tmp.items():
        if metrics_rw.get(k, None) == None:
            metrics_rw[k] = []
        metrics_rw[k].append(v)

        if metrics[edge_found].get(k, None) == None:
            metrics[edge_found][k] = []
        metrics[edge_found][k].append(v)

    # ===RS Sim===
    print('started sim rs')
    true_wug, simulation_wug_rs, max_iter_flag = simulation(true_wug, simulation_wug_rs, max_iter=500,
                        sampling_strategy=page_rank, sampling_params={'sample_size': 10, 'start': simulation_wug_rs.get_last_added_node, 'tp_coef': 1.0},
                        #  clustering_strategy=correlation_clustering, clustering_params={},
                        stopping_criterion=number_edges_found,  stopping_params={'number_edges': edge_found})
    print('ended sim rs')

    print('clustering rs')
    simulation_wug_rs.update_community_nodes_membership(correlation_clustering(simulation_wug_rs, {'weights': 'edge_soft_weight'}))
    print('clustering done rs')

    draw_graph_graphviz(simulation_wug_rs, save_flag=True, path='data/figs/other/sim_graph_{}_edges_{}_n{}_k{}.png'.format(title_rs, edge_found, nodes, com))

    tmp = analyze(true_wug, simulation_wug_rs, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), inverse_jensen_shannon_distance=(inverse_jensen_shannon_distance, {}))

    for k, v in tmp.items():
        if metrics_rs.get(k, None) == None:
            metrics_rs[k] = []
        metrics_rs[k].append(v)

        if metrics[edge_found].get(k, None) == None:
            metrics[edge_found][k] = []
        metrics[edge_found][k].append(v)
    

with open('data/metric_pr.data', 'wb') as file:
    pickle.dump(metrics_pr, file)
file.close()


with open('data/metric_rw.data', 'wb') as file:
    pickle.dump(metrics_rw, file)
file.close()

with open('data/metric_rs.data', 'wb') as file:
    pickle.dump(metrics_rs, file)
file.close()

with open('data/metrics.data', 'wb') as file:
    pickle.dump(metrics, file)
file.close()

for edge in edges:
    bar_metric(['PageRank', 'RandomWalk', 'RandomSampling'], 'Bar Graph of sim', 'Performance', True, 'data/figs/other/bar_plot_{}_n{}_k{}.png'.format(edge, nodes, com),
        adj_randIndex=metrics[edge]['adjusted_randIndex'], purity=metrics[edge]['purity'], accuracy=metrics[edge]['accuracy'], jensenshannon=metrics[edge]['inverse_jensen_shannon_distance'])

line_ploter(edges, 'Simulated Graph with PageRank', 'Edges', 'Performance', True, 'data/figs/other/line_plot_{}_n{}_k{}.png'.format(title_pr, nodes, com), 
    adj_randIndex=metrics_pr['adjusted_randIndex'], purity=metrics_pr['purity'], accuracy=metrics_pr['accuracy'], jensenshannon=metrics_pr['inverse_jensen_shannon_distance'])

line_ploter(edges, 'Simulated Graph with RandomWalk', 'Edges', 'Performance', True, 'data/figs/other/line_plot_{}_n{}_k{}.png'.format(title_rw, nodes, com), 
    adj_randIndex=metrics_rw['adjusted_randIndex'], purity=metrics_rw['purity'], accuracy=metrics_rw['accuracy'], jensenshannon=metrics_rw['inverse_jensen_shannon_distance'])

line_ploter(edges, 'Simulated Graph with RandomSampling', 'Edges', 'Performance', True, 'data/figs/other/line_plot_{}_n{}_k{}.png'.format(title_rs, nodes, com), 
    adj_randIndex=metrics_rs['adjusted_randIndex'], purity=metrics_rs['purity'], accuracy=metrics_rs['accuracy'], jensenshannon=metrics_rs['inverse_jensen_shannon_distance'])