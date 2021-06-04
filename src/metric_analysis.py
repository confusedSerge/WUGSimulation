import pickle

metric_data_path = 'data/graphs/sim_graphs/dwug/2021_06_04_20_52/metric/DWUG.data'

with open(metric_data_path, 'rb') as file:
    dwug_metric = pickle.load(file)
file.close()

print('done')