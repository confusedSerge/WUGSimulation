import pickle

from simulation.full_simulation import full_simulation

from graphs.wu_graph_sampler import WUGraphSampler
from graphs.wu_graph import WUGraph
from graphs.wu_simulation_graph import WUSimulationGraph

from simulation.sampling.sampling_strategy import page_rank
from simulation.sampling.sampling_strategy import dwug_sampling
from simulation.clustering.clustering_strategy import new_correlation_clustering
from simulation.stopping.stopping_criterion import number_edges_found
from simulation.stopping.stopping_criterion import edges_added
from simulation.stopping.stopping_criterion import cluster_connected

from analysis.analyzer import analyze
from analysis.metrics import *

from visualization.graph_visualization import draw_graph_graphviz
from visualization.metric_vis import threed_line_ploter

title_0 = 'vs'
title_pr = 'pageRank'
title_rw = 'randomWalk'
title_rs = 'randomSample'
title_dwug = 'dwug'
save_path_plot = 'data/figs/other/test'

# Graph params
nodes = 100
lognorm = 0.99
com = [1, 3, 5, 10]

# other
edge_found = 4000
split_flag = False
judgements = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]

# metric collector
metrics_dict_pr = {'adjusted_randIndex': {}, 'purity': {}, 'accuracy': {}, 'inverse_jensen_shannon_distance': {}}
metrics_dict_rw = {'adjusted_randIndex': {}, 'purity': {}, 'accuracy': {}, 'inverse_jensen_shannon_distance': {}}
metrics_dict_rs = {'adjusted_randIndex': {}, 'purity': {}, 'accuracy': {}, 'inverse_jensen_shannon_distance': {}}
metrics_dict_dwug = {'adjusted_randIndex': {}, 'purity': {}, 'accuracy': {}, 'inverse_jensen_shannon_distance': {}}


true_wug_gen = WUGraphSampler(nodes, com, ('log', {'std_dev': 0.99}), ['binomial', 3, 0.99]).sample_wug_generator()

for i, true_wug in enumerate(true_wug_gen):
    draw_graph_graphviz(true_wug, plot_title='True Graph', save_flag=True, path='{}/true_graph/k{}/{}_n{}_k{}.png'.format(save_path_plot, com[i], title_0, nodes, com[i]))

    # simulation_wug_pr = WUSimulationGraph(true_wug.get_number_nodes())
    # simulation_wug_rw = WUSimulationGraph(true_wug.get_number_nodes())
    # simulation_wug_rs = WUSimulationGraph(true_wug.get_number_nodes())
    simulation_wug_dwug = WUSimulationGraph(true_wug.get_number_nodes())

    # print('New K-Sim {}'.format(com[i]))

    # print('Started PageRank')
    # # pageRank
    # simulation_wug_pr, max_iter_pr, metrics_pr, metrics_pr_graphs \
    #     = full_simulation(trueGraph=true_wug, simulationGraph=simulation_wug_pr, max_iter=5000, verbose=True,
    #                 sampling_strategy=page_rank, sampling_params={'sample_size': 10, 'start': simulation_wug_pr.get_last_added_node, 'tp_coef': 0.1},
    #                 # clustering_strategy=correlation_clustering, clustering_params={},
    #                 stopping_criterion=number_edges_found,  stopping_params={'number_edges': edge_found},
    #                 analyzing_critertion=edges_added, analyzing_critertion_params=[{'number_edges': x} for x in judgements],
    #                 anal_clustering_strategy=new_correlation_clustering, anal_clustering_params={'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': split_flag},
    #                 analyzing_func=analyze, 
    #                 analyzing_params={'adjusted_randIndex': (adjusted_randIndex, {}), 'purity':(purity, {}), 'accuracy':(accuracy, {}), 'inverse_jensen_shannon_distance':(inverse_jensen_shannon_distance, {})}, 
    #                 return_graph_flag=True)

    # draw_graph_graphviz(simulation_wug_pr, plot_title='PageRank Final', save_flag=True, path='{}/pw/k{}/final_pr_sim_graph_{}_n{}_k{}.png'.format(save_path_plot, com[i], title_pr, nodes, com[i]))

    # for j, _graph in enumerate(metrics_pr_graphs):
    #     draw_graph_graphviz(_graph, plot_title='PageRank with {} judgements'.format(judgements[j]), save_flag=True, path='{}/pw/k{}/pr_sim_graph_{}_judg_{}_n{}_k{}.png'.format(save_path_plot, com[i], title_pr, judgements[j], nodes, com[i]))

    # for tmp_item in metrics_pr:
    #     for k, v in tmp_item.items():
    #         if metrics_dict_pr[k].get(com[i], None) == None:
    #             metrics_dict_pr[k][com[i]] = []
    #         metrics_dict_pr[k][com[i]].append(v)

    # print('Started RandomWalk')
    # # randomWalk
    # simulation_wug_rw, max_iter_rw, metrics_rw, metrics_rw_graphs \
    #     = full_simulation(trueGraph=true_wug, simulationGraph=simulation_wug_rw, max_iter=5000, verbose=True,
    #                 sampling_strategy=page_rank, sampling_params={'sample_size': 10, 'start': simulation_wug_rw.get_last_added_node, 'tp_coef': 0.0},
    #                 # clustering_strategy=correlation_clustering, clustering_params={},
    #                 stopping_criterion=number_edges_found,  stopping_params={'number_edges': edge_found},
    #                 analyzing_critertion=edges_added, analyzing_critertion_params=[{'number_edges': x} for x in judgements],
    #                 anal_clustering_strategy=new_correlation_clustering, anal_clustering_params={'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': split_flag},
    #                 analyzing_func=analyze, 
    #                 analyzing_params={'adjusted_randIndex': (adjusted_randIndex, {}), 'purity':(purity, {}), 'accuracy':(accuracy, {}), 'inverse_jensen_shannon_distance':(inverse_jensen_shannon_distance, {})}, 
    #                 return_graph_flag=True)

    # draw_graph_graphviz(simulation_wug_rw, plot_title='RandomWalk Final', save_flag=True, path='{}/rw/k{}/final_sim_graph_{}_n{}_k{}.png'.format(save_path_plot, com[i], title_rw, nodes, com[i]))

    # for j, _graph in enumerate(metrics_rw_graphs):
    #     draw_graph_graphviz(_graph, plot_title='RandomWalk with {} judgements'.format(judgements[j]), save_flag=True, path='{}/rw/k{}/sim_graph_{}_judg_{}_n{}_k{}.png'.format(save_path_plot, com[i],title_rw, judgements[j], nodes, com[i]))

    # for tmp_item in metrics_rw:
    #     for k, v in tmp_item.items():
    #         if metrics_dict_rw[k].get(com[i], None) == None:
    #             metrics_dict_rw[k][com[i]] = []
    #         metrics_dict_rw[k][com[i]].append(v)

    # print('Started RandomSampling')
    # # randomSampling
    # simulation_wug_rs, max_iter_rs, metrics_rs, metrics_rs_graphs \
    #     = full_simulation(trueGraph=true_wug, simulationGraph=simulation_wug_rs, max_iter=5000, verbose=True,
    #                 sampling_strategy=page_rank, sampling_params={'sample_size': 10, 'start': simulation_wug_rs.get_last_added_node, 'tp_coef': 1.0},
    #                 # clustering_strategy=correlation_clustering, clustering_params={},
    #                 stopping_criterion=number_edges_found,  stopping_params={'number_edges': edge_found},
    #                 analyzing_critertion=edges_added, analyzing_critertion_params=[{'number_edges': x} for x in judgements],
    #                 anal_clustering_strategy=new_correlation_clustering, anal_clustering_params={'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': split_flag},
    #                 analyzing_func=analyze, 
    #                 analyzing_params={'adjusted_randIndex': (adjusted_randIndex, {}), 'purity':(purity, {}), 'accuracy':(accuracy, {}), 'inverse_jensen_shannon_distance':(inverse_jensen_shannon_distance, {})}, 
    #                 return_graph_flag=True)

    # draw_graph_graphviz(simulation_wug_rs, plot_title='RandomSampling Final', save_flag=True, path='{}/rs/k{}/final_sim_graph_{}_n{}_k{}.png'.format(save_path_plot, com[i], title_rs, nodes, com[i]))

    # for j, _graph in enumerate(metrics_rs_graphs):
    #     draw_graph_graphviz(_graph, plot_title='RandomSampling with {} judgements'.format(judgements[j]), save_flag=True, path='{}/rs/k{}/sim_graph_{}_judg_{}_n{}_k{}.png'.format(save_path_plot, com[i], title_rs, judgements[j], nodes, com[i]))

    # for tmp_item in metrics_rs:
    #     for k, v in tmp_item.items():
    #         if metrics_dict_rs[k].get(com[i], None) == None:
    #             metrics_dict_rs[k][com[i]] = []
    #         metrics_dict_rs[k][com[i]].append(v)

    print('Started DWUG')
    simulation_wug_dwug, max_iter_rs, metrics_dwug, metrics_dwug_graphs \
        = full_simulation(trueGraph=true_wug, simulationGraph=simulation_wug_dwug, max_iter=5000, verbose=True,
                    sampling_strategy=dwug_sampling, sampling_params={'simulationGraph': simulation_wug_dwug, 'percentage_nodes': 0.1, 'percentage_edges': 0.3, 'min_size_mc': 2},
                    clustering_strategy=new_correlation_clustering, clustering_params={'weights': 'edge_soft_weight', 'max_attempts': 10, 'max_iters': 10, 'split_flag': split_flag},
                    stopping_criterion=cluster_connected,  stopping_params={'cluster_min_size': 1, 'min_num_edges': 5},
                    analyzing_critertion=edges_added, analyzing_critertion_params=[{'number_edges': x} for x in judgements],
                    analyzing_func=analyze, 
                    analyzing_params={'adjusted_randIndex': (adjusted_randIndex, {}), 'purity':(purity, {}), 'accuracy':(accuracy, {}), 'inverse_jensen_shannon_distance':(inverse_jensen_shannon_distance, {})}, 
                    return_graph_flag=True)

    draw_graph_graphviz(simulation_wug_dwug, plot_title='RandomSampling Final for k={}'.format(com[i]), save_flag=True, path='{}/dwug/k{}/final_sim_graph_{}_n{}_k{}.png'.format(save_path_plot, com[i], title_rs, nodes, com[i]))

    for j, _graph in enumerate(metrics_dwug_graphs):
        draw_graph_graphviz(_graph, plot_title='RandomSampling with {} judgements'.format(judgements[j]), save_flag=True, path='{}/dwug/k{}/sim_graph_{}_judg_{}_n{}_k{}.png'.format(save_path_plot, com[i], title_rs, judgements[j], nodes, com[i]))

    for tmp_item in metrics_dwug:
        for k, v in tmp_item.items():
            if metrics_dict_dwug[k].get(com[i], None) == None:
                metrics_dict_dwug[k][com[i]] = []
            metrics_dict_dwug[k][com[i]].append(v)
    
with open('data/metric_pr_wosplit.data', 'wb') as file:
    pickle.dump(metrics_dict_pr, file)
file.close()

with open('data/metric_rw_wosplit.data', 'wb') as file:
    pickle.dump(metrics_dict_rw, file)
file.close()

with open('data/metric_rs_wosplit.data', 'wb') as file:
    pickle.dump(metrics_dict_rs, file)
file.close()

with open('data/metric_dwug_wosplit.data', 'wb') as file:
    pickle.dump(metrics_dict_dwug, file)
file.close()

# threed_line_ploter(judgements, '#Judgements', com, '#Communities', 'Performance', 'Simulation PageRank', legend_flag=True,
#     adj_randIndex=metrics_dict_pr['adjusted_randIndex'], purity=metrics_dict_pr['purity'], accuracy=metrics_dict_pr['accuracy'], jensenshannon=metrics_dict_pr['inverse_jensen_shannon_distance'])

# threed_line_ploter(judgements, '#Judgements', com, '#Communities', 'Performance', 'Simulation PageRank', 
#     adj_randIndex=metrics_dict_pr['adjusted_randIndex'], purity=metrics_dict_pr['purity'], accuracy=metrics_dict_pr['accuracy'], jensenshannon=metrics_dict_pr['inverse_jensen_shannon_distance'])

# for k, v in metrics_dict_dwug.items():
#     for _k, _v in v.items():
#         _v.extend([_v[-1]] * (len(judgements) - len(_v))) 

# threed_line_ploter(judgements, '#Judgements', com, '#Communities', 'Performance', 'Simulation DWUG', legend_flag=True, 
#     adj_randIndex=metrics_dict_dwug['adjusted_randIndex'], purity=metrics_dict_dwug['purity'], accuracy=metrics_dict_dwug['accuracy'], jensenshannon=metrics_dict_dwug['inverse_jensen_shannon_distance'])