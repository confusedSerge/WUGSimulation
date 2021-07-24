import os
import pickle

from graphs.base_graph import BaseGraph
from graphs.annotated_graph import AnnotatedGraph

from simulation.simulation import Simulation
from simulation.sampling.sampling import Sampling
from simulation.sampling.sampling_strategy import page_rank
from simulation.sampling.annotator import Annotator
from simulation.stopping.stopping import Stopping
from simulation.stopping.stopping_criterion import number_edges_found
from simulation.utils.intermediate_save_listener import IntermediateSaveListener

path_true = 'data/graphs/kw29/bigdata/true'
path_out = 'data/graphs/kw29/bigdata/sim/randomsampling/{}'
# get all graphs paths
paths_to_true = []
for _, _, files in os.walk(path_true):
    file : str
    for file in files:
        if file.endswith('.graph'):
            paths_to_true.append((file, '{}/{}'.format(path_true, file)))
paths_to_true.sort()

# Run Sim
checkpoints = list(range(10, 100, 10))
checkpoints.extend(list(range(100, 5100, 100)))
name : str
path : str
for i, (name, path) in enumerate(paths_to_true):
    print("Graph {}: name {}".format(i + 1, name))
    with open(path, 'rb') as file:
        graph : BaseGraph = pickle.load(file)
    file.close()

    annotated_graph = AnnotatedGraph(graph.get_number_nodes())
    
    sampling_step = Sampling().add_sampling_strategie(page_rank, {'sample_size': 10, 'start': annotated_graph.get_last_added_node, 'tp_coef': 1.0})

    listener = IntermediateSaveListener().add_listener(checkpoints, path_out.format(name.replace('.graph', '')), '{}_j'.format(name.replace('.graph', '')), annotated_graph.get_num_added_edges)

    simulation = Simulation(1000, break_on_sc=False).add_step(sampling_step).add_step(listener)
    simulation.run(graph, annotated_graph)
