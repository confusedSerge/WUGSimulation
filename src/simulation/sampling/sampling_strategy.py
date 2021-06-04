import random
import numpy as np

from graphs.base_graph import BaseGraph
from graphs.wu_annotator_graph import WUAnnotatorGraph
from graphs.wu_annotator_simulation_graph import WUAnnotatorSimulationGraph

from simulation.sampling.utils.dwug_sampling import dwug_sampling as u_dwug_sampling

"""
This module contains different sampling function and can be extended to new ones.
All method signatures should look like this:
    def name(tG: TrueGraph, params: dict) -> list:
Each sampling function should return a list of sampled edges
"""


def random_sampling(trueGraph: BaseGraph, params: dict) -> list:
    """
    Random sampling. 
    As described in TACL paper 'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'

    This implementation takes n radom edges (:sample_size:) from the TrueGraph and returns it.

    Args:
        :param trueGraph: TrueGraph to sample
        :param sample_size: number of edges to sample
        :return sampled_edge_list: sampled edges with weights as [(u, v, w)...]
    
    """
    assert isinstance(trueGraph, BaseGraph)

    sample_size = params.get('sample_size', None)
    assert sample_size != None and type(sample_size) == int

    sampled_edge_list = []

    for i in range(sample_size):
        u, v = sorted(random.sample(trueGraph.G.nodes(), 2))
        sampled_edge_list.append((u, v, trueGraph.get_edge(u, v)))

    return sampled_edge_list

def page_rank(trueGraph: BaseGraph, params: dict) -> list:
    """
    Page Rank sampling strategy with equal transition probability.

    Important to note:
        - tp_coef == 1: Random Sample
        - tp_coef == 0: Random Walk

    Args:
        :param trueGraph: TrueGraph to sample
        :param sample_size: number of edges to sample per annotator
        :param start: start node (can be None, int, or function)
        :param tp_coef: teleportation coefficient

    """
    # ===Guard===
    sample_size = params.get('sample_size', None)
    assert type(sample_size) == int and sample_size > 0

    tp_coef = params.get('tp_coef', None)
    assert type(tp_coef) == float and 0 <= tp_coef <= 1

    last_node = params.get('start', None)
    if callable(last_node):
        last_node = last_node()

    assert type(last_node) == int or last_node == None 

    if last_node == None:
        last_node = random.sample(trueGraph.G.nodes(), 1)[0]
    # ===END Guard===

    sampled_edge_list = []

    for i in range(sample_size):
        # choose next start and following node
        last_node = np.random.choice([last_node, random.sample(trueGraph.G.nodes(), 1)[0]], p=[1 - tp_coef, tp_coef])
        next_node = random.sample(trueGraph.G.nodes(), 1)[0]

        sampled_edge_list.append((last_node, next_node, trueGraph.get_edge(last_node, next_node)))
        last_node = next_node

    return sampled_edge_list

def page_rank_per_annotator(trueGraph: WUAnnotatorGraph, params: dict) -> list:
    """
    Page Rank sampling strategy with equal transition probability

    Important: This is a specialized function for WUG simulation

    Args:
        :param trueGraph: WUAnnotatorGraph to sample
        :param simGraph: WUAnnotatorSimulationGraph
        :param sample_size: number of edges to sample per annotator
        :param tp_coef: teleportation coefficient
    """
    # ===Guard===
    assert isinstance(trueGraph, WUAnnotatorGraph)

    simGraph = params.get('simGraph', None)
    assert isinstance(simGraph, WUAnnotatorSimulationGraph)

    sample_size = params.get('sample_size', None)
    assert type(sample_size) == int and sample_size > 0

    tp_coef = params.get('tp_coef', None)
    assert type(tp_coef) == float and 0 <= tp_coef <= 1
    # ===END Guard===

    for i in range(trueGraph.get_num_annotators()):
        edge_list = page_rank(trueGraph, 
            {'sample_size': sample_size, 'tp_coef': tp_coef, 'start': simGraph.get_last_added_node(i)})
        for j in range(len(edge_list)):
            new_weight = trueGraph.get_edge(*edge_list[j][:2], annotator=i, add_prob=0.5)
            edge_list[j] = (*edge_list[j][:2], new_weight)
        simGraph.add_edges(edge_list, annotator=i)
    
    return None

def page_rank_across_annotators(trueGraph: WUAnnotatorGraph, params: dict) -> list:
    """
    Page Rank sampling strategy with equal transition probability

    Important: This is a specialized function for WUG simulation

    Args:
        :param trueGraph: WUAnnotatorGraph to sample
        :param simGraph: WUAnnotatorSimulationGraph
        :param sample_size: list with number of edges per annotator
        :param tp_coef: teleportation coefficient
    """
    # ===Guard===
    assert isinstance(trueGraph, WUAnnotatorGraph)

    simGraph = params.get('simGraph', None)
    assert isinstance(simGraph, WUAnnotatorSimulationGraph)

    sample_size = params.get('sample_size', None)
    assert type(sample_size) == list and sum(sample_size) > 0

    assert len(sample_size) == trueGraph.get_num_annotators()

    tp_coef = params.get('tp_coef', None)
    assert type(tp_coef) == float and 0 <= tp_coef <= 1
    # ===END Guard===

    edge_list = page_rank(trueGraph, 
        {'sample_size': sum(sample_size), 'tp_coef': tp_coef, 'start': simGraph.get_last_added_node()})

    start = 0
    for i in range(len(sample_size)):
        end = start + sample_size[i]
        new_edge_list = []
        for j in range(start, end):
            new_weight = trueGraph.get_edge(*edge_list[j][:2], annotator=i, add_prob=0.5)
            new_edge_list.append((*edge_list[j][:2], new_weight))
        start = end
        simGraph.add_edges(new_edge_list, annotator=i)

    return None

def dwug_sampling(trueGraph: BaseGraph, params: dict) -> list:
    """
    Uses the DWUG sampling strategy, as described in the paper.

    Args:
        :param trueGraph: graph on which to sample edge weight
        :param simulationGraph: simulation graph
        :param percentage_nodes: percentage of nodes to add this round
        :param percentage_edges: percentage of edges to add this round
        :param min_size_mc: minimum size of cluster to be considered as multi-cluster
    """
    # ===Guard Phase===
    assert isinstance(trueGraph, BaseGraph)

    simulationGraph = params.get('simulationGraph', None)
    assert simulationGraph != None and isinstance(simulationGraph, BaseGraph)

    percentage_nodes = params.get('percentage_nodes', None)
    assert type(percentage_nodes) == float and 0.0 <= percentage_nodes <= 1.0

    percentage_edges = params.get('percentage_edges', None)
    assert type(percentage_edges) == float and 0.0 <= percentage_edges <= 1.0

    min_size_mc = params.get('min_size_mc', None)
    assert type(min_size_mc) == int
    # ===Guard Phase over===

    return u_dwug_sampling(trueGraph, simulationGraph, percentage_nodes, percentage_edges, min_size_mc)
