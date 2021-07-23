import pickle
from simulation.clustering.clustering_strategy import new_correlation_clustering 

from graphs.base_graph import BaseGraph
from visualization.graph_visualization import draw_graph_graphviz as draw


steps = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
for j in steps:
    print('j={}'.format(j))
    path = 'data/graphs/sim_graphs/randomsampling/sim_ks_logsoft/2021_07_09_15_27/intermediate/k5/randomsampling_n100_k5_{}.graph'.format(j)
    tmp_path = 'data/figs/tmp'

    with open(path, 'rb') as file:
        graph : BaseGraph = pickle.load(file)
    file.close()

    _cl = new_correlation_clustering(graph, {'weights': 'edge_soft_weight', 'split_flag': True})
    print(_cl)
    graph.update_community_nodes_membership(_cl)
    draw(graph, "With islands", save_flag=True, path='{}/{}_wi'.format(tmp_path, j))

    _cl = new_correlation_clustering(graph, {'weights': 'edge_soft_weight', 'split_flag': False})
    print(_cl)
    graph.update_community_nodes_membership(_cl)
    draw(graph, "Without islands", save_flag=True, path='{}/{}_woi'.format(tmp_path, j))
