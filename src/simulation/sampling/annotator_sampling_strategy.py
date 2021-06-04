from graphs.wu_annotator_graph import WUAnnotatorGraph
from graphs.wu_annotator_simulation_graph import WUAnnotatorSimulationGraph
from simulation.sampling.sampling_strategy import page_rank

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

def dwug_sampling_per_annotator(trueGraph: WUAnnotatorGraph, params: dict) -> list:
    """
    DWUG sampling per annotator. This will perform dwug sampling per annotator with given parameter.

    Important: This is a specialized function for WUG simulation

    Args:
        :param trueGraph: WUAnnotatorGraph to sample
        :param simGraph: WUAnnotatorSimulationGraph
        :param percentage_nodes: percentage of nodes (per annotator) to add this round
        :param percentage_edges: percentage of edges (per annotator) to add this round
        :param min_size_mc: minimum size of cluster to be considered as multi-cluster
    """
     # ===Guard===
    assert isinstance(trueGraph, WUAnnotatorGraph)

    simGraph = params.get('simGraph', None)
    assert isinstance(simGraph, WUAnnotatorSimulationGraph)

    sample_size = params.get('sample_size', None)
    assert type(sample_size) == int and sample_size > 0
    pass


def dwug_sampling_across_annotators(trueGraph: WUAnnotatorGraph, params: dict) -> list:
    """
    docstring
    """
    pass