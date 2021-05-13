from visualization.nx_graph_plot import GraphDrawer
from true_graph.true_graph_sampler import TrueGraphSampler
from simulation.simulation import *
from simulation.sampling_strategy import random_sampling
from simulation.clustering_strategy import correlation_clustering
from simulation.stopping_criterion import percentage_edges_found
from visualization.community_distributions import community_distribution
from analysis.analyzer import analyze
from analysis.metrics import *

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


graph = TrueGraphSampler(100, 3, ('log', [0.99]), [
                         'binomial', 3, 0.9]).sample_graph()
drawer = GraphDrawer()


print('started sim')
tG, sG, max_iter_flag = simulation(graph, max_iter=250,
                    sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
                    #  clustering_strategy=correlation_clustering, clustering_params={},
                    stopping_criterion=percentage_edges_found,  stopping_params={'percentage': 0.05, 'number_edges': len(graph.get_nx_graph_rep().edges)})
print('ended sim')

print('clustering')
sG.update_community_membership(correlation_clustering(sG, {}))
print('clustering done')

print(max_iter_flag)
print(analyze(tG, sG, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), jensen_shannon_distance=(jensen_shannon_distance, {})))
drawer.draw_graph(sG.G)

print('started sim')
tG, sG, max_iter_flag = simulation(graph, max_iter=250,
                    sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
                    #  clustering_strategy=correlation_clustering, clustering_params={},
                    stopping_criterion=percentage_edges_found,  stopping_params={'percentage': 0.1, 'number_edges': len(graph.get_nx_graph_rep().edges)})
print('ended sim')

print('clustering')
sG.update_community_membership(correlation_clustering(sG, {}))
print('clustering done')

print(max_iter_flag)
print(analyze(tG, sG, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), jensen_shannon_distance=(jensen_shannon_distance, {})))
drawer.draw_graph(sG.G)

print('started sim')
tG, sG, max_iter_flag = simulation(graph, max_iter=250,
                    sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
                    #  clustering_strategy=correlation_clustering, clustering_params={},
                    stopping_criterion=percentage_edges_found,  stopping_params={'percentage': 0.2, 'number_edges': len(graph.get_nx_graph_rep().edges)})
print('ended sim')

print('clustering')
sG.update_community_membership(correlation_clustering(sG, {}))
print('clustering done')

print(max_iter_flag)
print(analyze(tG, sG, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), jensen_shannon_distance=(jensen_shannon_distance, {})))
drawer.draw_graph(sG.G)

print('started sim')
tG, sG, max_iter_flag = simulation(tG=graph, sG=sG, max_iter=250,
                    sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
                    #  clustering_strategy=correlation_clustering, clustering_params={},
                    stopping_criterion=percentage_edges_found,  stopping_params={'percentage': 0.5, 'number_edges': len(graph.get_nx_graph_rep().edges)})
print('ended sim')

print('clustering')
sG.update_community_membership(correlation_clustering(sG, {}))
print('clustering done')

print(max_iter_flag)
print(analyze(tG, sG, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), jensen_shannon_distance=(jensen_shannon_distance, {})))

drawer.draw_graph(sG.G)

print('started sim')
tG, sG, max_iter_flag = simulation(tG=graph, sG=sG, max_iter=500,
                    sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
                    #  clustering_strategy=correlation_clustering, clustering_params={},
                    stopping_criterion=percentage_edges_found,  stopping_params={'percentage': 0.7, 'number_edges': len(graph.get_nx_graph_rep().edges)})

print('ended sim')

print('clustering')
sG.update_community_membership(correlation_clustering(sG, {}))
print('clustering done')

print(max_iter_flag)
print(analyze(tG, sG, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), jensen_shannon_distance=(jensen_shannon_distance, {})))

drawer.draw_graph(sG.G)

print('started sim')
tG, sG, max_iter_flag = simulation(tG=graph, sG=sG, max_iter=500,
                    sampling_strategy=random_sampling, sampling_params={'sample_size': 10},
                    #  clustering_strategy=correlation_clustering, clustering_params={},
                    stopping_criterion=percentage_edges_found,  stopping_params={'percentage': 0.9, 'number_edges': len(graph.get_nx_graph_rep().edges)})
print('ended sim')

print('clustering')
sG.update_community_membership(correlation_clustering(sG, {}))
print('clustering done')


print(max_iter_flag)
print(analyze(tG, sG, adjusted_randIndex=(adjusted_randIndex, {}), purity=(purity, {}), accuracy=(accuracy, {}), jensen_shannon_distance=(jensen_shannon_distance, {})))

drawer.draw_graph(sG.G)


# drawer.draw_graph(tG.get_nx_graph_rep(), save_flag=True, path='data/figs/other/tG.png')
# drawer.draw_graph(sG.G, save_flag=True, path='data/figs/other/sG.png')
