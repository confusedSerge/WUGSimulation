import os
import pickle
import numpy as np

from graphs.fitted_graph import FittedGraph

from visualization.graph_visualization import draw_graph_gt as draw


path = 'data/graphs/all_float_dict'
path_out = 'experiment_data/fitted_graphs'
path_draw = 'data/graphs/all_float_dict_fig'

for _, _, files in os.walk(path):
    for dict_file in files:
        print('=================================')
        print('Working on {}'.format(dict_file))

        with open('{}/{}'.format(path, dict_file), 'rb') as file:
            dist_dict: dict = pickle.load(file)
        file.close()

        graph = FittedGraph(dist_dict)
        draw(graph, '{}/{}.png'.format(path_draw, dict_file.replace('.distribution', '')))

        print('Writing Graph to {}'.format('{}/{}.graph'.format(path_out, dict_file.replace('.distribution', ''))))
        with open('{}/{}.graph'.format(path_out, dict_file.replace('.distribution', '')), 'wb') as file:
            pickle.dump(graph, file)
        file.close()
# draw(graph)
