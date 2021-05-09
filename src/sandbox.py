from visualization.nx_graph_plot import GraphDrawer
from true_graph.true_graph_sampler import TrueGraphSampler
from simulation.simulation import *
from simulation.sampling_strategy import random_sampling
from simulation.clustering_strategy import correlation_clustering
from simulation.stopping_criterion import cluster_connected
from visualization.community_distributions import community_distribution

# graph_generator = TrueGraphSampler(
#     100, [*range(1, 11)], ('log', [0.99]), ['binomial', 3, 0.9]).sample_graph_generator()
# drawer = GraphDrawer()

# results = simulation_with_tG_generator(graph_generator, max_iter=100,
#  sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
#  clustering_strategy=correlation_clustering, clustering_params={},
#  stopping_criterion=cluster_connected,  stopping_params={'cluster_min_size': 5, 'min_num_edges': 'fully'})

# for tG, sG in results:
#     drawer.draw_graph(tG.get_nx_graph_rep())
#     print(sG)
#     drawer.draw_graph(sG.G)


graph = TrueGraphSampler(100, 3, ('log', [0.99]), ['binomial', 3, 0.9]).sample_graph()
drawer = GraphDrawer()

community_distribution(graph.get_nx_graph_rep())

tG, sG = simulation(graph, max_iter=100,
 sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
 clustering_strategy=correlation_clustering, clustering_params={},
 stopping_criterion=cluster_connected,  stopping_params={'cluster_min_size': 5, 'min_num_edges': 'fully', 'min_size_one_cluster': 75})

drawer.draw_graph(tG.get_nx_graph_rep(), save_flag=True, path='data/figs/other/tG.png')
print(sG)
drawer.draw_graph(sG.G, save_flag=True, path='data/figs/other/sG.png')