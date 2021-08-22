import os
import pickle
import numpy as np
from graphs.base_graph import BaseGraph


def load_graph_from_path_file(path_list: list):
    _pl = np.array(path_list)
    _graphs = np.empty(_pl.shape, dtype=BaseGraph)
    it = np.nditer(np.zeros(_pl.shape), flags=['multi_index'])
    for _ in it:
        path = _pl[it.multi_index]

        with open(path, 'rb') as file:
            _graphs[it.multi_index] = pickle.load(file)
        file.close()
    return _graphs


def load_graph_from_path_dir(path_list: list, suffix, sort_func, shape):
    _pl = np.array(path_list, dtype=str)
    _graphs = np.empty(shape, dtype=BaseGraph)
    it = np.nditer(np.zeros(_pl.shape), flags=['multi_index'])
    for _ in it:
        path = str(_pl[it.multi_index])

        graphs = []
        for _, _, files in os.walk(path):
            for file in files:
                if file.endswith(suffix):
                    with open('{}/{}'.format(path, file), 'rb') as fg:
                        graphs.append(pickle.load(fg))
                    fg.close()
        graphs.sort(key=sort_func)
        if len(graphs) != shape[-1]:
            graphs.extend([graphs[-1]] * (shape[-1] - len(graphs)))
        _graphs[it.multi_index[:2]] = np.array(graphs)
    return _graphs
