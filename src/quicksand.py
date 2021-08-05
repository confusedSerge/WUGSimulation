import numpy as np
import random
from itertools import combinations
import pickle

from graphs.simulation_graph_sampler import SimulationGraphSampler
from simulation.stopping.stopping_criterion import bootstraping_jsd

n = 100
k = 3
log = 0.9
size_communities = ('log_iter', {'std_dev': log, 'threshold': 5})
distribution = ['binomial', 3, 0.99]

# graph = SimulationGraphSampler(n, k, size_communities, distribution).sample_simulation_graph()
# # for i in range(n):
# #     print(bootstraping_jsd(true, {'min_sample_size': 100, 'rounds': 30, 'sample_size': 150, 'alpha': 0.95, 'bound': 0.05}))
