# # import random
# from networkx.algorithms.components import node_connected_component

# from true_graph.true_graph_sampler import TrueGraphSampler
# from simulation.sampling_strategy import random_sampling
# from simulation.simulation_graph import SimulationGraph

# # from simulation.clustering_strategy import correlation_clustering

# # # graph_generator = TrueGraphSampler(100, [*range(1, 11)], ('log', [0.99]), ['binomial', 3, 0.9]).sample_graph_generator()

# # # drawer = gp.GraphDrawer()
# # # for graph in graph_generator:
# # #     nx_graph = graph.get_nx_graph_rep()
# # #     drawer.draw_graph(nx_graph)

# graph = TrueGraphSampler(100, 3, ('log', [0.99]), ['binomial', 3, 0.9]).sample_graph()
# sG = SimulationGraph()

# # # drawer = gp.GraphDrawer()

# # # nx_graph = graph.get_nx_graph_rep()
# # # drawer.draw_graph(nx_graph)

# for i in range(1):
#     edge_list = random_sampling(graph, sample_size=10)
#     # print(edge_list)
#     sG.add_edges(edge_list)

# # clusters = correlation_clustering(sG)

blus = [(0, 1, 1), (0, 2, 3), (0, 1, 3), (1, 0, 4)]
bla = {}
blas = {}

for u, v, w in blus: 
    print(u, v, w)
    u, v = sorted([u, v])
    bla[(u, v)] = w 

for k, v in bla.items():
    blas[v] = k

print(len(bla), bla)
print(len(blas), blas)