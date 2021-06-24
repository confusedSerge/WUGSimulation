import pickle
import os
import logging

from datetime import datetime

from graphs.wu_graph import WUGraph
from graphs.wu_simulation_graph import WUSimulationGraph

from simulation.full_simulation import full_simulation
from simulation.sampling.sampling_strategy import page_rank
from simulation.clustering.clustering_strategy import new_correlation_clustering
from simulation.stopping.stopping_criterion import edges_added
from simulation.stopping.stopping_criterion import number_edges_found

from analysis.metric_results import MetricResults
from analysis.analyzer import analyze
from analysis.metrics import *

from visualization.graph_visualization import draw_graph_graphviz as draw

"""
This is a special script for running random sampling sims.
"""
# === Setup Phase ===
# Note: Change these accordingly
sim_short_name, sim_property_name = 'RandomSampling', 'sim_ks_loghard'

# vars for simulation
# Note: settings of sim also check/change function
sort_func = lambda x: x.get_number_communities()

# drawer and pickle vars 
plot_title, plot_name = 'Random Sampling Simulation', 'randomsampling_sim'

# other vars
save_intermediate, draw_intermediate = True, True
verbose = True
path_true_wugs = 'data/graphs/true_graphs/k_c_var/2021_06_11_10_36'

#sim param (some can only be init in sim loop, but for completness add here as string, so it can be logged)
judgments_points = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

max_iter, break_on_sc, verbose = 5000, False, verbose,

sampling_strategy, sampling_params = page_rank, {'sample_size': 10, 'start': 'simulation_wug.get_last_added_node', 'tp_coef': 1.0}
stopping_criterion, stopping_params = number_edges_found, {'number_edges': 4000}
clustering_strategy, clustering_params = None, 'None'

analyzing_critertion, analyzing_critertion_params = edges_added, [{'number_edges': x} for x in judgments_points]
anal_clustering_strategy, anal_clustering_params = new_correlation_clustering, {'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': False}
analyzing_func=analyze 
analyzing_params={'adjusted_randIndex': (adjusted_randIndex, {}), 'purity':(purity, {}), 'accuracy':(accuracy, {}), 'inverse_jensen_shannon_distance':(inverse_jensen_shannon_distance, {})}

return_graph_flag=True

# === More or less automated process ===
now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

path_graph_plot = 'data/figs/sim_graphs/{}/{}/{}'.format(sim_short_name.lower(), sim_property_name, current_time)
path_graph_save = 'data/graphs/sim_graphs/{}/{}/{}'.format(sim_short_name.lower(), sim_property_name, current_time)
path_metric_save = 'data/graphs/sim_graphs/{}/{}/{}'.format(sim_short_name.lower(), sim_property_name,current_time)

try:
    os.makedirs(path_graph_save)
except FileExistsError as identifier:
    logging.error('Something went wrong with the creation of the base path!')
    logging.error(identifier)
    exit()

logging.basicConfig(filename='{}/sim.log'.format(path_graph_save),format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# Config Logging
logging.log(21, 'Logging various Parameters')
logging.log(21, 'sim_short_name, sim_property_name = {}, {}'.format(sim_short_name, sim_property_name))
logging.log(21, 'sort_func = {}'.format(sort_func.__name__))
logging.log(21, 'plot_title, plot_name = {}, {}'.format(plot_title, plot_name))
logging.log(21, 'save_intermediate, draw_intermediate = {}, {}'.format(save_intermediate, draw_intermediate))
logging.log(21, 'verbose = {}'.format(verbose))
logging.log(21, 'path_true_wugs = {}'.format(path_true_wugs))

logging.log(21, 'Logging Parameters of Simulation')
logging.log(21, 'max_iter, break_on_sc, verbose = {}, {}, {}'.format(max_iter, break_on_sc, verbose))

logging.log(21, 'sampling_strategy, sampling_params = {}, {}'.format(sampling_strategy.__name__, sampling_params))
logging.log(21, 'stopping_criterion, stopping_params = {}, {}'.format(stopping_criterion.__name__, stopping_params))
logging.log(21, 'clustering_strategy, clustering_params = {}, {}'.format(clustering_strategy.__name__ if clustering_strategy != None else 'None', clustering_params))

logging.log(21, 'analyzing_critertion, analyzing_critertion_params = {}, {}'.format(analyzing_critertion.__name__, analyzing_critertion_params))
logging.log(21, 'anal_clustering_strategy, anal_clustering_params = {}, {}'.format(anal_clustering_strategy.__name__, anal_clustering_params))
logging.log(21, 'analyzing_func, analyzing_params = {}, {}'.format(analyzing_func.__name__, analyzing_params))
logging.log(21, 'return_graph_flag = {}'.format(return_graph_flag))
logging.log(21, 'End Logging Parameters')

# === True Graph Phase === 
# Load true_graph/s
if verbose: logging.info('Loading graphs from: {}'.format(path_true_wugs))
graphs = []
for _, _, files in os.walk(path_true_wugs):
    for file in files:
        if file.endswith('.graph'):
            with open('{}/{}'.format(path_true_wugs, file), 'rb') as fg:
                graphs.append(pickle.load(fg))
            fg.close()

# Check, that all loaded items are graphs
if verbose: logging.info('Cleaning loaded graphs')
for i, graph in enumerate(graphs):
    try:
        assert isinstance(graph, WUGraph)
    except AssertionError as identifier:
       graphs.pop(i)
       if verbose: logging.info('Removed item, as it is not an instance of WUGraph')

if verbose: logging.info('Ordering graphs')
graphs.sort(key=sort_func)


# === mkdir Phase ===
if verbose: logging.info('Making Directories')
try:
    os.makedirs('{}/final'.format(path_graph_plot))
    if verbose: logging.info('Path created: {}/final'.format(path_graph_plot))
except FileExistsError as identifier:
    if verbose: logging.error('Path creation failed: {}/final'.format(path_graph_plot))

try:
    os.makedirs('{}/intermediate'.format(path_graph_plot))
    if verbose: logging.info('Path created: {}/intermediate'.format(path_graph_plot))
except FileExistsError as identifier:
    if verbose: logging.error('Path creation failed: {}/intermediate'.format(path_graph_plot))

try:
    os.makedirs('{}/final'.format(path_graph_save))
    if verbose: logging.info('Path created: {}/final'.format(path_graph_save))
except FileExistsError as identifier:
    if verbose: logging.error('Path creation failed: {}/final'.format(path_graph_save))

try:
    os.makedirs('{}/intermediate'.format(path_graph_save))
    if verbose: logging.info('Path created: {}/intermediate'.format(path_graph_save))
except FileExistsError as identifier:
    if verbose: logging.error('Path creation failed: {}/intermediate'.format(path_graph_save))

try:
    os.makedirs('{}/metric'.format(path_metric_save))
    if verbose: logging.info('Path created: {}/metric'.format(path_metric_save))
except FileExistsError as identifier:
    if verbose: logging.error('Path creation failed: {}/metric'.format(path_metric_save))

# === Metric Class Phase ===
# create Metric Result object 
# Note: need to be adjusted for sims
_metric_result = MetricResults(sim_short_name, 'Created at: {}'.format(current_time))
for _metric in ['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance']:
    _metric_result.add_metric(_metric, (len(graphs), len(judgments_points)), ('ks = {}'.format([graph.get_number_communities() for graph in graphs]), 'judgement points'))


# === Sim Phase ===
if verbose: logging.info('Starting SIMs')
for i, graph in enumerate(graphs):
    simulation_wug = WUSimulationGraph(graph.get_number_nodes())

    if verbose: logging.info('Started {} Simulation with k={}'.format(sim_short_name, graph.get_number_communities()))
    # === Simulation. Vars need to be adjusted here ===
    simulation_wug, max_iter_hit, _metrics, _metric_graphs \
        = full_simulation(trueGraph=graph, simulationGraph=simulation_wug, 
                    max_iter=max_iter, break_on_sc=break_on_sc, verbose=verbose,
                    
                    sampling_strategy=sampling_strategy, sampling_params={'sample_size': 10, 'start': simulation_wug.get_last_added_node, 'tp_coef': 1.0},
                    stopping_criterion=stopping_criterion,  stopping_params=stopping_params,
                    
                    analyzing_critertion=analyzing_critertion, analyzing_critertion_params=analyzing_critertion_params,
                    anal_clustering_strategy=anal_clustering_strategy, anal_clustering_params=anal_clustering_params,
                    analyzing_func=analyzing_func, analyzing_params=analyzing_params, 
                    
                    return_graph_flag=return_graph_flag)
    if verbose: 
        logging.info('Finished {} Simulation with k={}'.format(sim_short_name, graph.get_number_communities()))
        logging.info('Max Iterations hit: {}'.format(max_iter_hit))

    # === Paths Creation & saving ===
    path_plot_final_sim = '{}/final/{}_n{}_k{}'.format(path_graph_plot, plot_name, graph.get_number_nodes(), graph.get_number_communities())
    path_graph_final_sim = '{}/final/{}_n{}_k{}'.format(path_graph_save, plot_name, graph.get_number_nodes(), graph.get_number_communities())
    path_plot_intermediate_sims = ['{}/intermediate/k{}/{}_n{}_k{}_j{}'.format(path_graph_plot, graph.get_number_communities(), plot_name, graph.get_number_nodes(), graph.get_number_communities(), judgement) for judgement in judgments_points]
    path_graph_intermediate_sims = ['{}/intermediate/k{}/{}_n{}_k{}_j{}'.format(path_graph_save, graph.get_number_communities(), plot_name, graph.get_number_nodes(), graph.get_number_communities(), judgement) for judgement in judgments_points]

    # saving draws and graph data 
    draw(simulation_wug, plot_title='{} Final for k={}'.format(plot_title, graph.get_number_communities()), save_flag=True, path='{}.png'.format(path_plot_final_sim))
    if verbose: logging.info('Final {}-SIM drawn at: {}.png'.format(sim_short_name, path_plot_final_sim))

    with open('{}.graph'.format(path_graph_final_sim), 'wb') as file:
        pickle.dump(simulation_wug, file)
    file.close()
    if verbose: logging.info('Final {}-SIM saved at: {}.graph'.format(sim_short_name, path_graph_final_sim))

    for j, _graph in enumerate(_metric_graphs):
        # mkdir for intermediate/k
        if draw_intermediate:
            try:
                os.makedirs('{}/intermediate/k{}'.format(path_graph_plot, graph.get_number_communities()))
                if verbose: logging.info('Path created: {}/intermediate/k{}'.format(path_graph_plot, graph.get_number_communities()))
            except FileExistsError as identifier:
                pass

            draw(_graph, plot_title='{} for k = {} with {} judgments'.format(plot_title, graph.get_number_communities(), judgments_points[j]),
                save_flag=True, path='{}.png'.format(path_plot_intermediate_sims[j]))
            if verbose: logging.info('Intermediate {}-SIM drawn at: {}.png'.format(sim_short_name, path_plot_intermediate_sims[j]))

        if save_intermediate:
            try:
                os.makedirs('{}/intermediate/k{}'.format(path_graph_save, graph.get_number_communities()))
                if verbose: logging.info('Path created: {}/intermediate/k{}'.format(path_graph_save, graph.get_number_communities()))
            except FileExistsError as identifier:
                pass

            with open('{}.graph'.format(path_graph_intermediate_sims[j]), 'wb') as file:
                pickle.dump(_graph, file)
            file.close()
            if verbose: logging.info('Intermediate {}-SIM saved at: {}.graph'.format(sim_short_name, path_graph_intermediate_sims[j]))

    # === Saving Metric data in Metric Result Class ===
    if verbose: logging.info('Adding Results to Metric Class')
    tmp_metric = {}
    for tmp_item in _metrics:
        for k, v in tmp_item.items():
            if tmp_metric.get(k, None) == None:
                tmp_metric[k] = []
            tmp_metric[k].append(v)
    if verbose: logging.info(tmp_metric)
    for k, v in tmp_metric.items():
        _metric_result.update_value(k, v, i)

# === Saving metrics ===
if verbose: logging.info('Saving Metric Class in: {}/metric/{}.data'.format(path_metric_save, _metric_result.name))
with open('{}/metric/{}.data'.format(path_metric_save, _metric_result.name), 'wb') as file:
    pickle.dump(_metric_result, file)
file.close()

if verbose: logging.info('Finished')

# === Save Info === 
