import pickle
from visualization.metric_vis import threed_line_ploter

with open('data/metric_pr.data', 'rb') as file:
    metrics_dict = pickle.load(file)
file.close()

edges = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
nodes = 100
com = [*range(1, 11)]


threed_line_ploter(edges, '#Edges', com, '#Communities', 'Performance', 'Simulation PageRank', False,
    adj_randIndex=metrics_dict['adjusted_randIndex'], purity=metrics_dict['purity'], accuracy=metrics_dict['accuracy'], jensenshannon=metrics_dict['inverse_jensen_shannon_distance'])