import os
import pickle

from graphs.simulation_graph_sampler import SimulationGraphSampler
from datetime import datetime

n = 100
ks = [1, 3, 5, 7, 10]
logs = [0.1, 0.3, 0.5, 0.7, 0.9]
distribution = ['binomial', 3, 0.99]
num_trues = 10

path_out = 'data/graphs/kw29/bigdata/true'
name = 'n{}_k{}_log{}_{}.graph'

try:
    os.makedirs(path_out)
except FileExistsError:
    pass

for k in ks:
    for log in logs:
        for i in range(num_trues):
            size_communities = ('log_iter', {'std_dev': log, 'threshold': 5})

            true = SimulationGraphSampler(n, k, size_communities, distribution).sample_simulation_graph()
            with open('{}/{}'.format(path_out, name.format(n, k, log, i)), 'wb') as file:
                pickle.dump(true, file)
            file.close()