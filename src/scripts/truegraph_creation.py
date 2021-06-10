import os
import pickle
import csv
import logging

from graphs.wu_graph_sampler import WUGraphSampler
from datetime import datetime

"""
This is a special script for running some sims.
"""

# pre init
now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")
path = 'data/graphs/true_graphs/{}'.format(current_time)

try:
    os.makedirs(path)
except FileExistsError as identifier:
    logging.error('Something went wrong with the creation of the base path!')
    logging.error(identifier)
    exit()

logging.basicConfig(filename='{}/graph_gen.log'.format(path),format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger('graph_gen_logger')
logger.setLevel(logging.INFO)

# Graph params
nodes = 100
com = [1, 3, 5, 10]

# size_communities = ('log_iter', {'std_dev': 0.1, 'threshold': 5})
size_communities = ('log_iter', {'std_dev': 0.9, 'threshold': 5})
csv_c = 'log_iter(0.1, 5)'
name_c = 'log_iter_0.9_5'

distribution = ['binomial', 3, 0.99]
csv_d = 'binomial(3 0.99)'
name_d = 'binomial_3_0.99'

annotators = None
csv_a = 'None'
name_a = 'none'

true_wug_names = ['true_graph_wug_n{}_k{}_c{}_d{}_a{}.graph'.format(nodes, k, name_c, name_d, name_a) for k in com]

# other
verbose_flag = True
multiple_flag = type(com) == list

# save graph/s
if multiple_flag:
    true_wug_gen = WUGraphSampler(nodes, com, size_communities, distribution, annotators).sample_wug_generator()

    for i, tg in enumerate(true_wug_gen):
        with open('{}/{}'.format(path, true_wug_names[i]), 'wb') as file:
            pickle.dump(tg, file)
        file.close()
        if verbose_flag: logger.info('True Graph saved: {}/{}'.format(path, true_wug_names[i]))

else:
    true_wug_gen = WUGraphSampler(nodes, com, size_communities, distribution, annotators).sample_wug()
    with open('{}/{}'.format(path, true_wug_names[0]), 'wb') as file:
            pickle.dump(tg, file)
    file.close()
    if verbose_flag: logger.info('True Graph saved: {}/{}'.format(path, true_wug_names[0]))

# gen and save info about graph/s as csv
csv_array = [['n(odes)', 'k(ommunity)', 'c(ommunity size)', 'd(istdribution)', 'a(nnotators)']]

if multiple_flag:
    for k in com:
        csv_array.append([nodes, k, csv_c, csv_d, csv_a])
else:
    csv_array.append([nodes, com, csv_c, csv_d, csv_a])

with open('{}/info_param.csv'.format(path), 'w') as file:
    writer = csv.writer(file) 
    writer.writerows(csv_array)
file.close()

if verbose_flag: logger.info('Parameter-File written: {}/info_param.csv'.format(path))