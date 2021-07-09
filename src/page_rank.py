import os
import pickle
import numpy as np
from datetime import datetime

from graphs.base_graph import BaseGraph
from graphs.annotated_graph import AnnotatedGraph

from simulation.simulation import Simulation
from simulation.sampling.sampling import Sampling
from simulation.sampling.sampling_strategy import page_rank
from simulation.sampling.annotator import Annotator
from simulation.stopping.stopping import Stopping
from simulation.stopping.adv_stopping_criterion import entropy_approx_convergence
from simulation.stopping.adv_stopping_criterion import entropy_approx_convergence_reset
from simulation.utils.intermediate_save_listener import IntermediateSaveListener

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

path_sim_graphs = 'data/graphs/true_graphs/k_c_var/2021_07_09_13_13'
out_path = 'data/graphs/sim_graphs/pagerank/sim_ks_loghard'
sort_func = lambda x: x.get_number_communities()


# Load Graphs
graphs = []
for _, _, files in os.walk(path_sim_graphs):
    for file in files:
        if file.endswith('.graph'):
            with open('{}/{}'.format(path_sim_graphs, file), 'rb') as fg:
                graphs.append(pickle.load(fg))
            fg.close()

# Check, that all loaded items are graphs
for i, graph in enumerate(graphs):
    try:
        assert isinstance(graph, BaseGraph)
    except AssertionError as identifier:
       graphs.pop(i)

graphs.sort(key=sort_func)

# Run Sim
for graph in graphs:
    graph : BaseGraph = graph
    annotated_graph = AnnotatedGraph(graph.get_number_nodes())
    
    sampling_step = Sampling().add_sampling_strategie(page_rank, {'sample_size': 10, 'start': annotated_graph.get_last_added_node, 'tp_coef': 0.1})\
        .add_annotator(Annotator(np.random.poisson, [0.2], 1, 4, 0.5))\
        .add_annotator(Annotator(np.random.poisson, [0.2], 1, 4, 0.5))\
        .add_annotator(Annotator(np.random.poisson, [0.2], 1, 4, 0.5))\
        .add_annotator(Annotator(np.random.poisson, [0.2], 1, 4, 0.5))\
        .add_annotator(Annotator(np.random.poisson, [0.2], 1, 4, 0.5))\
        .set_annotator_dist('random')

    stopping_step = Stopping().add_adv_stopping_criterion(entropy_approx_convergence, {'threshold_conv': 0.02}, entropy_approx_convergence_reset)
    
    checkpoints = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
    path_intermediate_graph = '{}/{}/intermediate/k{}'.format(out_path, current_time, graph.get_number_communities())
    listener = IntermediateSaveListener().add_listener(checkpoints, path_intermediate_graph, 'pagerank_n{}_k{}'.format(graph.get_number_nodes(), graph.get_number_communities()), annotated_graph.get_num_added_edges)

    simulation = Simulation(1000).add_step(sampling_step).add_step(listener).add_step(stopping_step)
    simulation.run(graph, annotated_graph)

    final_graph : BaseGraph = stopping_step.sc_annotated_graph 
    sc_hit: bool = stopping_step.sc_hit_flag
    final_graph.G.graph['sc_hit'] = sc_hit

    path_out = '{}/{}/final'.format(out_path, current_time)
    try:
        os.makedirs(path_out)
    except FileExistsError:
        pass

    with open('{}/pagerank_n{}_k{}.graph'.format(path_out, graph.get_number_nodes(), graph.get_number_communities()), 'wb') as file:
        pickle.dump(final_graph, file)
    file.close()
