import random
import numpy as np

from graphs.base_graph import BaseGraph

"""
This module contains different simple sampling functions and can be extended to new ones.
All method signatures should look like this:
    def name(graph: BaseGraph, params: dict) -> list:
Each sampling function should return a list of sampled edges
"""


def random_sampling(graph: BaseGraph, params: dict) -> list:
    """
    Random sampling. 
    As described in TACL paper 'Word Usage Graphs (WUGs):Measuring Changes in Patterns of Contextual Word Meaning'

    This implementation takes n radom edges (:sample_size:) from the TrueGraph and returns it.

    Args:
        :param trueGraph: TrueGraph to sample
        :param sample_size: number of edges to sample
        :return sampled_edge_list: sampled edges with weights as [(u, v, w)...]
    
    """
    assert isinstance(graph, BaseGraph)

    sample_size = params.get('sample_size', None)
    assert sample_size != None and type(sample_size) == int

    sampled_edge_list = []

    for _ in range(sample_size):
        u, v = sorted(random.sample(graph.G.nodes(), 2))
        sampled_edge_list.append((u, v, graph.get_edge(u, v)))

    return sampled_edge_list

def page_rank(graph: BaseGraph, params: dict) -> list:
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
        :return sampled_edge_list: sampled edges with weights as [(u, v, w)...]

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
        last_node = random.sample(graph.G.nodes(), 1)[0]
    # ===END Guard===

    sampled_edge_list = []

    for _ in range(sample_size):
        # choose next start and following node
        last_node = np.random.choice([last_node, random.sample(graph.G.nodes(), 1)[0]], p=[1 - tp_coef, tp_coef])
        next_node = random.sample(graph.G.nodes(), 1)[0]

        sampled_edge_list.append((last_node, next_node, graph.get_edge(last_node, next_node)))
        last_node = next_node

    return sampled_edge_list
