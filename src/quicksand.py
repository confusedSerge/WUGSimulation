# just a script to play around, test some functionalities, find bugs
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

from visualization.graph_visualization import draw_graph_graphviz as draw
import pickle
import os

path_true_wugs = 'data/graphs/true_graphs/2021_06_10_14_53'

graphs = []
for _, _, files in os.walk(path_true_wugs):
    for file in files:
        if file.endswith('.graph'):
            with open('{}/{}'.format(path_true_wugs, file), 'rb') as fg:
                graphs.append(pickle.load(fg))
            fg.close()

for i, graph in enumerate(graphs):
        draw(graph, plot_title='True Graph', save_flag=True, path='data/figs/{}.png'.format(i))