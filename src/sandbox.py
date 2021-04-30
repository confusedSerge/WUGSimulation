import graph_sim as gs
import graph_plot as gp
from datetime import datetime

gss = gs.GraphSimulatorSamplerFactory().add_ul_bound_nodes(50, 50).add_ul_bound_communities(2, 3).add_dist_binomial(3, 3, 0.9, 0.9).build_graph_simulator_sampler()

for _ in range(5):
    graph, info = gss.sample_graph(True)

    number_nodes, number_communities, node_com, distribution = info
    now = datetime.now()
    stmp = now.strftime("%Y%m%d%H%M%S%f")

    graph.save_graph('./data/graph_pickle/nodes_{}_communities_{}_bin_{}_{}_stmp_{}'.format(number_nodes, number_communities, distribution.tries, distribution.probability, stmp))

    drawer = gp.GraphDrawer()
    drawer.draw_graph_from_graph_sim(graph, save_flag=True, path='./data/figs/nodes_{}_communities_{}_bin_{}_{}_stmp_{}.png'.format(number_nodes, number_communities, distribution.tries, distribution.probability, stmp))


