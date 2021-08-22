import numpy as np

from graphs.wu_graph_sampler import WUGraphSampler
from graphs.wu_simulation_graph import WUSimulationGraph

from simulation.full_simulation import full_simulation
from simulation.sampling.sampling_strategy import page_rank
from simulation.stopping.stopping_criterion import number_edges_found
from simulation.stopping.stopping_criterion import edges_added
from simulation.clustering.clustering_strategy import new_correlation_clustering

from analysis.metric_results import MetricResults
from analysis.analyzer import analyze
from analysis.metrics import *

from visualization.graph_visualization import draw_graph_graphviz as draw
from visualization.metric_vis import boxplot_metric_pd as boxplot
from visualization.metric_vis import bar_metric_pd as barplot
from visualization.metric_vis import heatmap

# If you want to try this graph, it is recommended to put this script in the top directory 

# WARNING: This does not work anymore, as the simulation logic has changed drastically!!

'''
Generating a new Graph can be done through the Graph Class itself, or through a sampler,
from which multiple graphs can be generated. 
By using a sampler, it is possible to use different functions, like the log_norm function to generate the cluster sizes.
'''
true_wug = WUGraphSampler(100, 3, ('log', {'std_dev': 0.99}), ['binomial', 3, 0.99]).sample_wug()

'''
The visualization module provides multiple functions to plot graphs as well as metrics.
'''
draw(true_wug, 'True Graph')

'''
The BaseGraph provides an interface (as well as some pre-defined functions, which can be overwritten) 
for all graphs, which are functions the framework expects to be implemented.
This also provides customizability, as the backbone implementation can be changed to fit the experiment. 
The framework only cares about the correct results delivered by the called function.
'''
simulation_wug_rs = WUSimulationGraph(true_wug.get_number_nodes())

'''
The MetricResults-Class provides an CRUD interface for dealing with metrics collected during simulation.
'''
metric_result_rs = MetricResults(name='Random Sampling Metrics')

'''
We provide multiple simulation functions, but recommend the full_simulation function.
In every iteration of the simulation 4 steps are performed: Sampling, Clustering (optional), Analysing (optional), checking Stopping Criterion.
The analysis step also contains an optional clustering, which is only performed if the Clustering step is skipped and a clustering algorithm is provided. 
Also it is important to note, that the analysis step is only performed for the given analysis points.  
For the simulation we provide the functions and parameters used for each step (sampling, clustering, analysing, stopping)
as well as a true graph (on which to sample) and a simulation graph.
The reason for providing a simulation graph is that it can either be a customized graph, or a graph from previous simulations.
We can also provide different analysing criterions as well as points at which we want to analyse the current simulation graph.
'''
# randomSampling
judgments = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]
simulation_wug_rs, max_iter_reached, metrics_rs, metrics_rs_graphs \
    = full_simulation(trueGraph=true_wug, simulationGraph=simulation_wug_rs, max_iter=5000, break_on_sc=False, verbose=False,
                sampling_strategy=page_rank, sampling_params={'sample_size': 10, 'start': simulation_wug_rs.get_last_added_node, 'tp_coef': 1.0},
                # clustering_strategy=new_correlation_clustering, clustering_params={'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': True},
                stopping_criterion=number_edges_found,  stopping_params={'number_edges': 4000},
                analyzing_critertion=edges_added, analyzing_critertion_params=[{'number_edges': x} for x in judgments],
                anal_clustering_strategy=new_correlation_clustering, anal_clustering_params={'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': True},
                analyzing_func=analyze, 
                analyzing_params={'adjusted_randIndex': (adjusted_randIndex, {}), 'purity':(purity, {}), 'accuracy':(accuracy, {}), 'inverse_jensen_shannon_distance':(inverse_jensen_shannon_distance, {})}, 
                return_graph_flag=True)

'''
If no clustering is performed (due to not providing any clustering strategies), it is recommended to do a clustering of the graph after the simulation
If clustering is only performed during the analysis step, it is also recommended to cluster the final simulation graph, as the final graph may not be clustered!
'''
simulation_wug_rs.update_community_nodes_membership(new_correlation_clustering(simulation_wug_rs, {'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': True}))

draw(simulation_wug_rs, 'Final Simulation Graph')

'''
Metrics collected during simulation can be either seen raw, 
as a list of dictionaries containing each metric at each pre-defined point,
or be converted and stored to the MetricResults-Object. 
'''
metrics_used = ['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance']
metric_result_rs.add_metric('all', (len(metrics_used), len(judgments)), ('0: metric', '1: judgement'))

for _metric in metrics_used:
    metric_result_rs.add_metric(_metric, (len(judgments)), ('0: judgments'))

tmp_metric = {}
for tmp_item in metrics_rs:
    for k, v in tmp_item.items():
        if tmp_metric.get(k, None) == None:
            tmp_metric[k] = []
        tmp_metric[k].append(v)

for i, (k, v) in enumerate(tmp_metric.items()):
    metric_result_rs.update_value(k, v)
    metric_result_rs.update_value('all', v, i)

'''
Of course the positive effects of using this data class structure only shows,
when multiple parameters are tested, as through this data structure it is possible to
single values or different slices of these values (like getting the values for certain fixed subset of parameters).
'''
print(metric_result_rs.get_values('all'))
print(metric_result_rs.get_values('inverse_jensen_shannon_distance', 0))


'''
The framework also includes different visualization functions for the generated data.
'''
boxplot('Boxplot of all data from Random Sampling', 'Performance', 
    adjusted_randIndex=metric_result_rs.get_values('adjusted_randIndex'), 
    purity=metric_result_rs.get_values('purity'), 
    accuracy=metric_result_rs.get_values('accuracy'), 
    inverse_jensen_shannon_distance=metric_result_rs.get_values('inverse_jensen_shannon_distance'))

barplot(judgments, 'Barplot of all data from Random Sampling', 'Performance', 
    adjusted_randIndex=metric_result_rs.get_values('adjusted_randIndex'), 
    purity=metric_result_rs.get_values('purity'), 
    accuracy=metric_result_rs.get_values('accuracy'), 
    inverse_jensen_shannon_distance=metric_result_rs.get_values('inverse_jensen_shannon_distance'))

heatmap(np.round(metric_result_rs.get_values('all').T, 2), 'Heatmap of Random Sampling', metrics_used, judgments)
