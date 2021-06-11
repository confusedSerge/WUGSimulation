import pickle
import os
import logging

from datetime import datetime

from analysis.metric_results import MetricResults

'''
Important, this is just a sandbox script for merging data 
'''

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

# Change var here
metric_name = 'DWUG k[1, 3, 5, 10] log[hard, soft]'
metric_short = 'dwug_comb'

path_hard = 'data/graphs/sim_graphs/dwug/sim_ks_loghard/2021_06_11_14_55/metric/DWUG.data'
path_soft = 'data/graphs/sim_graphs/dwug/sim_ks_logsoft/2021_06_11_14_55/metric/DWUG.data'
path_out = 'data/graphs/sim_graphs/dwug/sim_ks_logsofthard/{}'.format(current_time)

metrics = ['adjusted_randIndex', 'purity', 'accuracy', 'inverse_jensen_shannon_distance']

# ===Automatic process===
try:
    os.makedirs(path_out)
except FileExistsError as identifier:
    logging.error('Something went wrong with the creation of the base path!')
    logging.error(identifier)
    exit()

try:
    os.makedirs('{}/metric'.format(path_out))
except FileExistsError as identifier:
    logging.error('Something went wrong with the creation of the base path!')
    logging.error(identifier)
    exit()

# logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.basicConfig(filename='{}/sim.log'.format(path_out),format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

logging.info('Loading Metrics')
with open(path_hard, 'rb') as file:
    metric_loghard : MetricResults = pickle.load(file)
file.close() 

with open(path_soft, 'rb') as file:
    metric_logsoft : MetricResults = pickle.load(file)
file.close()

logging.info('Combining metrics')
metric_comb = MetricResults(metric_name, current_time)
for _metric in metrics:
    metric_comb.add_metric(_metric, (2, *metric_loghard.get_values(_metric).shape), ('log=[hard, soft]', *metric_loghard.get_axes_info(_metric)))
    metric_comb.update_value(_metric, metric_loghard.get_values(_metric), 0)
    metric_comb.update_value(_metric, metric_logsoft.get_values(_metric), 1)

logging.info('Saving combined metric as: {}/metric/{}.data'.format(path_out, metric_short))
with open('{}/metric/{}.data'.format(path_out, metric_short), 'wb') as file:
    pickle.dump(metric_comb, file)
file.close()

logging.info('Finished')