import pickle
# from visualization.metric_vis import threed_line_ploter

# with open('data/metric_pr.data', 'rb') as file:
#     metrics_dict = pickle.load(file)
# file.close()

# edges = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
# nodes = 100
# com = [*range(1, 11)]


# threed_line_ploter(edges, '#Edges', com, '#Communities', 'Performance', 'Simulation PageRank', False,
#     adj_randIndex=metrics_dict['adjusted_randIndex'], purity=metrics_dict['purity'], accuracy=metrics_dict['accuracy'], jensenshannon=metrics_dict['inverse_jensen_shannon_distance'])

# from matplotlib import pyplot as plt
import numpy as np
# from scipy.stats import lognorm

# std_devs = [0.99, 0.25]

# num_nodes = 100
# num_communities = 10

# for std_dev in std_devs:
#     community_split = lognorm.pdf(np.linspace(0, num_communities - 1, num_communities), std_dev) * num_nodes
#     print(community_split)
#     community_split = [int(x) if int(x) > 0 else 1 for x in community_split]

#     community_split[0] += num_nodes - sum(community_split)

#     print(std_dev)
#     print(community_split)
judgements = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]

print([{'number_edges': x} for x in judgements])

test = [1, 2, 3, 4, 5, 6]

try:
    print(np.random.choice(test, 7, replace=False))
except ValueError as identifier:
    print(test)