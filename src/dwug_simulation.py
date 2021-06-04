import pickle
import os

from datetime import datetime

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

from visualization.graph_visualization import draw_graph_graphviz as draw

# get currenttime
now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

# vars for simulation
# Note: settings of sim need to be set at function
judgements_anpoint = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
sort_func = lambda x: x.get_number_communities()

# drawer and pickle vars 
plot_title = 'DWUG Simulation'
plot_name = 'dwug_sim'

path_graph_plot = 'data/figs/sim_graphs/dwug/{}'.format(current_time)
path_graph_save = 'data/graphs/sim_graphs/dwug/{}'.format(current_time)
path_metric_save = 'data/graphs/sim_graphs/dwug/{}'.format(current_time)


# other vars
save_intermediate = True
draw_intermediate = True
verbose = True
path_true_wugs = 'data/graphs/true_graphs/2021_06_04_16_08'


# Load true_graph/s
if verbose: print('Loading graphs')
graphs = []
for _, _, files in os.walk(path_true_wugs):
    for file in files:
        if file.endswith('.graph'):
            with open('{}/{}'.format(path_true_wugs, file), 'rb') as fg:
                graphs.append(pickle.load(fg))
            fg.close()

# Check, that all loaded items are graphs
if verbose: print('Cleaning loaded graphs')
for i, graph in enumerate(graphs):
    try:
        assert isinstance(graph, WUGraph)
    except AssertionError as identifier:
       graphs.pop(i)
       if verbose: print('Removed item, as it is not an instance of WUGraph')

if verbose: print('Ordering graphs')
graphs.sort(key=sort_func)


# mkdirs
if verbose: print('Making Directories')
try:
    os.makedirs('{}/final'.format(path_graph_plot))
    if verbose: print('Path created: {}/final'.format(path_graph_plot))
except FileExistsError as identifier:
    # if verbose: print('Path creation failed: {}/final'.format(path_graph_plot))
    pass

try:
    os.makedirs('{}/intermediate'.format(path_graph_plot))
    if verbose: print('Path created: {}/intermediate'.format(path_graph_plot))
except FileExistsError as identifier:
    # if verbose: print('Path creation failed: {}/intermediate'.format(path_graph_plot))
    pass

try:
    os.makedirs('{}/final'.format(path_graph_save))
    if verbose: print('Path created: {}/final'.format(path_graph_save))
except FileExistsError as identifier:
    # if verbose: print('Path creation failed: {}/final'.format(path_graph_save))
    pass

try:
    os.makedirs('{}/intermediate'.format(path_graph_save))
    if verbose: print('Path created: {}/intermediate'.format(path_graph_save))
except FileExistsError as identifier:
    # if verbose: print('Path creation failed: {}/intermediate'.format(path_graph_save))
    pass

try:
    os.makedirs('{}/metric'.format(path_metric_save))
    if verbose: print('Path created: {}/metric'.format(path_metric_save))
except FileExistsError as identifier:
    # if verbose: print('Path creation failed: {}/intermediate'.format(path_graph_save))
    pass
# create Metric Result object
dwug_metrics = MetricResults('DWUG', 'Created at: {}'.format(current_time))
for _metric in ['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance']:
    dwug_metrics.add_metric(_metric, (len(graphs), len(judgements_anpoint)), ('ks = '.format([graph.get_number_communities() for graph in graphs]), 'judgement point'))


# # Sim
if verbose: print('Starting SIMs')
for i, graph in enumerate(graphs):
    # just for typing/intellisense to work, can be removed later
    graph: WUGraph = graph

    simulation_wug_dwug = WUSimulationGraph(graph.get_number_nodes())

    if verbose: print('Started DWUG Simulation with k={}'.format(graph.get_number_communities()))
    simulation_wug_dwug, max_iter_rs, metrics_dwug, metrics_dwug_graphs \
        = full_simulation(trueGraph=graph, simulationGraph=simulation_wug_dwug, max_iter=5000, break_on_sc=False, verbose=verbose,
                    sampling_strategy=dwug_sampling, sampling_params={'simulationGraph': simulation_wug_dwug, 'percentage_nodes': 0.1, 'percentage_edges': 0.3, 'min_size_mc': 2},
                    clustering_strategy=new_correlation_clustering, clustering_params={'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': False},
                    stopping_criterion=cluster_connected,  stopping_params={'cluster_min_size': 2, 'min_num_edges': 1},
                    analyzing_critertion=edges_added, analyzing_critertion_params=[{'number_edges': x} for x in judgements_anpoint],
                    analyzing_func=analyze, 
                    analyzing_params={'adjusted_randIndex': (adjusted_randIndex, {}), 'purity':(purity, {}), 'accuracy':(accuracy, {}), 'inverse_jensen_shannon_distance':(inverse_jensen_shannon_distance, {})}, 
                    return_graph_flag=True)
    if verbose: 
        print('Finished DWUG Simulation with k={}'.format(graph.get_number_communities()))
        print('Max Iterations hit: {}'.format(max_iter_rs))

    # TODO
    # paths
    path_plot_final_sim = '{}/final/{}_n{}_k{}'.format(path_graph_plot, plot_name, graph.get_number_nodes(), graph.get_number_communities())
    path_graph_final_sim = '{}/final/{}_n{}_k{}'.format(path_graph_save, plot_name, graph.get_number_nodes(), graph.get_number_communities())
    path_plot_intermediate_sims = ['{}/intermediate/k{}/{}_n{}_k{}_j{}'.format(path_graph_plot, graph.get_number_communities(), plot_name, graph.get_number_nodes(), graph.get_number_communities(), judgement) for judgement in judgements_anpoint]
    path_graph_intermediate_sims = ['{}/intermediate/k{}/{}_n{}_k{}_j{}'.format(path_graph_save, graph.get_number_communities(), plot_name, graph.get_number_nodes(), graph.get_number_communities(), judgement) for judgement in judgements_anpoint]

    # saving draws and graph data 
    draw(simulation_wug_dwug, plot_title='{} Final for k={}'.format(plot_title, graph.get_number_communities()), save_flag=True, path='{}.png'.format(path_plot_final_sim))
    if verbose: print('Final DWUG-SIM drawn at: {}.png'.format(path_plot_final_sim))

    with open('{}.graph'.format(path_graph_final_sim), 'wb') as file:
        pickle.dump(simulation_wug_dwug, file)
    file.close()
    if verbose: print('Final DWUG-SIM saved at: {}.graph'.format(path_graph_final_sim))

    for j, _graph in enumerate(metrics_dwug_graphs):
        # mkdir for intermediate/k
        if draw_intermediate:
            try:
                os.makedirs('{}/intermediate/k{}'.format(path_graph_plot, graph.get_number_communities()))
                if verbose: print('Path created: {}/intermediate/k{}'.format(path_graph_plot, graph.get_number_communities()))
            except FileExistsError as identifier:
                # if verbose: print('Path creation failed: {}/intermediate/k{}'.format(path_graph_plot, graph.get_number_communities()))
                pass

            draw(_graph, plot_title='{} for k = {} with {} judgements'.format(plot_title, graph.get_number_communities(), judgements_anpoint[j]),
                save_flag=True, path='{}.png'.format(path_plot_intermediate_sims[j]))
            if verbose: print('Intermediate DWUG-SIM drawn at: {}.png'.format(path_plot_intermediate_sims[j]))

        if save_intermediate:
            try:
                os.makedirs('{}/intermediate/k{}'.format(path_graph_save, graph.get_number_communities()))
                if verbose: print('Path created: {}/intermediate/k{}'.format(path_graph_save, graph.get_number_communities()))
            except FileExistsError as identifier:
                # if verbose: print('Path creation failed: {}/intermediate/k{}'.format(path_graph_save, graph.get_number_communities()))
                pass
            with open('{}.graph'.format(path_graph_intermediate_sims[j]), 'wb') as file:
                pickle.dump(simulation_wug_dwug, file)
            file.close()
            if verbose: print('Intermediate DWUG-SIM saved at: {}.graph'.format(path_graph_intermediate_sims[j]))

    # adding metric data
    if verbose: print('Adding Results to Metric Class')
    for tmp_item in metrics_dwug:
        for k, v in tmp_item.items():
            dwug_metrics.update_value(k, v, i)

# Saving metrics
if verbose: print('Saving Metric Class in: {}/metric/{}.data'.format(path_metric_save, dwug_metrics.name))
with open('{}/metric/{}.data'.format(path_metric_save, dwug_metrics.name), 'wb') as file:
    pickle.dump(dwug_metrics, file)
file.close()

if verbose: print('Finished')
