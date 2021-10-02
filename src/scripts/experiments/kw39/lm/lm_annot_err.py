import numpy as np

from graphs.simulation_graph import SimulationGraph
from graphs.utils.distribution import Binomial

from graphs.annotated_graph import AnnotatedGraph

from simulation.simulation import Simulation

from simulation.sampling.annotator import Annotator
from simulation.sampling.sampling import Sampling
from simulation.sampling.sampling_strategy import modified_randomwalk as mrw

from simulation.clustering.clustering import Clustering
from simulation.clustering.clustering_strategy import chinese_whisper_clustering as cw

from simulation.utils.intermediate_save_listener import IntermediateSaveListener

from visualization.graph_visualization import draw_graph_gt as draw

# without noise
graph = SimulationGraph([10, 10], None, Binomial(3, .9, 2))
draw(graph)

annotated_graph = AnnotatedGraph(20)

sampling = Sampling(annotations_per_edge=1).add_sampling_strategie(mrw, {'sample_size': 10, 'start': annotated_graph.get_last_added_node, 'conntained_func': annotated_graph.G.nodes})\
    .add_annotator(Annotator().add_error_sampling(np.random.poisson, dict(lam=0.35), 1, 4, 0.5).add_zero_probability(1 / 30).add_high_error_nodes([1, 2, 3], np.random.poisson, dict(lam=0.9), 1, 1, 0))\
    .set_annotator_dist('random')

clustering = Clustering().add_clustering_strategy(cw, {'weights': 'edge_soft_weight'})

listener = IntermediateSaveListener().add_listener([ii for ii in range(10, 600, 10)], 'data/graphs/kw39/sim/zero_and_annotator_test', 'small_test', annotated_graph.get_num_added_edges)\
    .save_draw()\
    .add_preprocessing_step(clustering)

simulation = Simulation(600, break_on_sc=False, verbose=True).add_step(sampling).add_step(listener)
simulation.run(graph, annotated_graph)
