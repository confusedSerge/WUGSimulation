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
    assert sample_size is not None and type(sample_size) == int

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

    assert type(last_node) == int or last_node is None

    if last_node is None:
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


def modified_randomwalk(graph: BaseGraph, params: dict) -> list:
    """This is a modified Randomwalk, which performs a random walk on the graph,
        but changes between the found nodes set and not found nodes set, from which the next destination is chosen.

        It is assumed, that the sample size is divisble by two!
    Args:
        graph (BaseGraph): Graph, on which the walk should be performed
        params (dict): Contains the parameters for sample_size, start_node, function that returns nodes of annotated graph

    Returns:
        list: edge list containing the the found edges
    """
    sample_size = params.get('sample_size', None)
    assert type(sample_size) == int and sample_size > 0
    sample_size = int(sample_size / 2)

    last_node = params.get('start', None)
    if callable(last_node):
        last_node = last_node()

    assert type(last_node) == int or last_node is None

    if last_node is None:
        last_node = random.sample(graph.G.nodes(), 1)[0]

    contained_set = params.get('conntained_func', None)
    assert callable(contained_set)
    contained_set = set(contained_set()).union({last_node})
    not_contained_set = set(graph.G.nodes()).difference(contained_set)

    sampled_edge_list = []

    for _ in range(sample_size):
        # Explore uknown set, if possible
        if len(not_contained_set) > 0:
            unknown_node = random.sample(not_contained_set, 1)[0]
        else:
            unknown_node = random.sample(contained_set.difference({last_node}), 1)[0]
        sampled_edge_list.append((last_node, unknown_node, graph.get_edge(last_node, unknown_node)))

        # return back to known set
        last_node = random.sample(contained_set.difference({unknown_node}), 1)[0]
        sampled_edge_list.append((unknown_node, last_node, graph.get_edge(unknown_node, last_node)))

        # add uknown node to known set
        not_contained_set = not_contained_set.difference({unknown_node})
        contained_set = contained_set.union({unknown_node})

    return sampled_edge_list


def multiple_edges_randomwalk(graph: BaseGraph, params: dict) -> list:
    """Random Walk with multiple edges sampled per visited node.
    A random sampled edge will be walked.

    The last edge in the list, symbolizes the last edge traversed.
    Rounds stands for how often edges will be traversed.

    Args:
        graph (BaseGraph): Graph to sample from
        params (dict): dict containing (rounds,  sample_per_node, start)

    Returns:
        list: sampled edges with weights [[u, v, w], ...]
    """
    # ===Guard===
    rounds = params.get('rounds', None)
    assert type(rounds) == int and rounds > 0

    sample_per_node = params.get('sample_per_node', None)
    assert type(sample_per_node) == int and sample_per_node > 0

    last_node = params.get('start', None)
    if callable(last_node):
        last_node = last_node()

    assert type(last_node) == int or last_node is None

    if last_node is None:
        last_node = random.sample(graph.G.nodes(), 1)[0]
    # ===END Guard===

    sampled_edge_list = []

    for _ in range(rounds):
        # choose next start and following node
        sampled_edge_list.extend([[last_node, next_node, graph.get_edge(last_node, next_node)] for next_node in random.sample(list(graph.G.nodes() - {last_node}), sample_per_node)])
        last_node = sampled_edge_list[-1][1]

    return sampled_edge_list
