import visualization.graph_plot as gp

from true_graph.true_graph_sampler import TrueGraphSampler

graph_generator = TrueGraphSampler(100, [*range(1, 11)], ('log', [0.99]), ['binomial', 3, 0.9]).sample_graph_generator()

drawer = gp.GraphDrawer()
for graph in graph_generator:
    nx_graph = graph.get_nx_graph_rep()
    drawer.draw_graph(nx_graph)

# graph = TrueGraphSampler((1, 1000), (1, 10), ('log', [0.99]), ['binomial', 3, 0.9]).sample_graph()
# drawer = gp.GraphDrawer()

# nx_graph = graph.get_nx_graph_rep()
# drawer.draw_graph(nx_graph)
