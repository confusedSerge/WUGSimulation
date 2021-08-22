import numpy as np
import random
from itertools import combinations
import pickle

from graphs.simulation_graph_sampler import SimulationGraphSampler
from simulation.stopping.stopping_criterion import bootstraping_jsd

from visualization.graph_visualization import draw_graph_graphviz as draw

with open('data/graphs/kw32/sim/randomsampling/n100_k1_log0.1-0-randomsampling_cc_nosplit/n100_k1_log0.1-0-randomsampling_cc_nosplit_j400.graph', 'rb') as file:
    graph = pickle.load(file)
file.close()

draw(graph, 'Test')