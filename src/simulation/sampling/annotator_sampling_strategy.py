from graphs.wu_annotator_graph import WUAnnotatorGraph
from graphs.wu_annotator_simulation_graph import WUAnnotatorSimulationGraph
from simulation.sampling.sampling_strategy import page_rank
from simulation.sampling.sampling_strategy import dwug_sampling

def page_rank_per_annotator(trueGraph: WUAnnotatorGraph, params: dict) -> list:
    """
    Page Rank sampling strategy with equal transition probability

    Important: This is a specialized function for WUG simulation

    Args:
        :param trueGraph: WUAnnotatorGraph to sample
        :param simulationGraph: WUAnnotatorSimulationGraph
        :param sample_size: number of edges to sample per annotator
        :param tp_coef: teleportation coefficient
        :return sampled_edge_list: sampled edges with weights as [(u, v, w)...]
    """
    # ===Guard===
    assert isinstance(trueGraph, WUAnnotatorGraph)

    simulationGraph = params.get('simulationGraph', None)
    assert isinstance(simulationGraph, WUAnnotatorSimulationGraph)

    sample_size = params.get('sample_size', None)
    assert type(sample_size) == int and sample_size > 0

    tp_coef = params.get('tp_coef', None)
    assert type(tp_coef) == float and 0 <= tp_coef <= 1
    # ===END Guard===

    for i in range(trueGraph.get_num_annotators()):
        edge_list = page_rank(trueGraph, 
            {'sample_size': sample_size, 'tp_coef': tp_coef, 'start': simulationGraph.get_last_added_node(i)})
        for j in range(len(edge_list)):
            new_weight = trueGraph.get_edge(*edge_list[j][:2], annotator=i, add_prob=0.5)
            edge_list[j] = (*edge_list[j][:2], new_weight)
        simulationGraph.add_edges(edge_list, annotator=i)
    
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
        :return sampled_edge_list: sampled edges with weights as [(u, v, w)...]
    """
    # ===Guard===
    assert isinstance(trueGraph, WUAnnotatorGraph)

    simulationGraph = params.get('simulationGraph', None)
    assert isinstance(simulationGraph, WUAnnotatorSimulationGraph)

    sample_size = params.get('sample_size', None)
    assert type(sample_size) == list and sum(sample_size) > 0

    assert len(sample_size) == trueGraph.get_num_annotators()

    tp_coef = params.get('tp_coef', None)
    assert type(tp_coef) == float and 0 <= tp_coef <= 1
    # ===END Guard===

    edge_list = page_rank(trueGraph, 
        {'sample_size': sum(sample_size), 'tp_coef': tp_coef, 'start': simulationGraph.get_last_added_node()})

    start = 0
    for i, sa in enumerate(sample_size):
        end = start + sample_size[i] if i < len(sample_size) - 1 else len(edge_list)
        new_edge_list = []
        for j in range(start, end):
            new_weight = trueGraph.get_edge(*edge_list[j][:2], annotator=i, add_prob=0.5)
            new_edge_list.append((*edge_list[j][:2], new_weight))
        start = end
        simulationGraph.add_edges(new_edge_list, annotator=i)

    return None

def dwug_sampling_per_annotator(trueGraph: WUAnnotatorGraph, params: dict) -> list:
    """
    DWUG sampling per annotator. This will perform dwug sampling per annotator with given parameter.

    Important: This is a specialized function for WUG simulation

    Args:
        :param trueGraph: WUAnnotatorGraph to sample
        :param simulationGraph: WUAnnotatorSimulationGraph
        :param percentage_nodes: percentage of nodes (per annotator) to add this round
        :param percentage_edges: percentage of edges (per annotator) to add this round
        :param min_size_mc: minimum size of cluster to be considered as multi-cluster
        :param num_flag: if :percentage_nodes: & :percentage_edges: are the actual number of nodes/edges to be used  (optional)
        :return sampled_edge_list: sampled edges with weights as [(u, v, w)...]
    """
     # ===Guard===
    assert isinstance(trueGraph, WUAnnotatorGraph)

    simulationGraph = params.get('simulationGraph', None)
    assert isinstance(simulationGraph, WUAnnotatorSimulationGraph)

    percentage_nodes = params.get('percentage_nodes', None)
    assert (type(percentage_nodes) == float and 0.0 <= percentage_nodes <= 1.0) or (type(percentage_nodes) == int and 0 <= percentage_nodes <= trueGraph.get_number_nodes())

    percentage_edges = params.get('percentage_edges', None)
    assert (type(percentage_edges) == float and 0.0 <= percentage_edges <= 1.0) or (type(percentage_edges) == int and 0 <= percentage_edges <= trueGraph.get_number_edges())
    
    min_size_mc = params.get('min_size_mc', None)
    assert type(min_size_mc) == int

    num_flag = params.get('num_flag', False)
    assert type(num_flag) == bool
    # ===End Guard===

    for i in range(trueGraph.get_num_annotators()):
        edge_list = dwug_sampling(trueGraph, {'simulationGraph': simulationGraph,
         'percentage_nodes': percentage_nodes, 'percentage_edges': percentage_edges,
         'min_size_mc': min_size_mc, 'num_flag': num_flag})
        for j in range(len(edge_list)):
            new_weight = trueGraph.get_edge(*edge_list[j][:2], annotator=i, add_prob=0.5)
            edge_list[j] = (*edge_list[j][:2], new_weight)
        simulationGraph.add_edges(edge_list, annotator=i)
    
    return None



def dwug_sampling_across_annotators(trueGraph: WUAnnotatorGraph, params: dict) -> list:
    """
    DWUG sampling across annotator. This will perform dwug sampling across annotator with given parameter.

    Important: This is a specialized function for WUG simulation

    Args:
        :param trueGraph: WUAnnotatorGraph to sample
        :param simulationGraph: WUAnnotatorSimulationGraph
        :param percentage_nodes: percentage of nodes to add this round
        :param percentage_edges: percentage of edges to add this round
        :param min_size_mc: minimum size of cluster to be considered as multi-cluster
        :param num_flag: if :percentage_nodes: & :percentage_edges: are the actual number of nodes/edges to be used  (optional)
        :param percentage_annotate: list of percentages, where percentage_annotate[i] describes how many edges annotator i annotates 
        :return sampled_edge_list: sampled edges with weights as [(u, v, w)...]
    """
    # ===Guard===
    assert isinstance(trueGraph, WUAnnotatorGraph)

    simulationGraph = params.get('simulationGraph', None)
    assert isinstance(simulationGraph, WUAnnotatorSimulationGraph)

    percentage_nodes = params.get('percentage_nodes', None)
    assert type(percentage_nodes) == list

    percentage_edges = params.get('percentage_edges', None)
    assert type(percentage_edges) == list

    percentage_nodes = params.get('percentage_nodes', None)
    assert (type(percentage_nodes) == float and 0.0 <= percentage_nodes <= 1.0) or (type(percentage_nodes) == int and 0 <= percentage_nodes <= trueGraph.get_number_nodes())

    percentage_edges = params.get('percentage_edges', None)
    assert (type(percentage_edges) == float and 0.0 <= percentage_edges <= 1.0) or (type(percentage_edges) == int and 0 <= percentage_edges <= trueGraph.get_number_edges())
    
    min_size_mc = params.get('min_size_mc', None)
    assert type(min_size_mc) == int

    num_flag = params.get('num_flag', False)
    assert type(num_flag) == bool

    percentage_annotate = params.get('percentage_annotate', False)
    assert type(percentage_annotate) == list and sum(percentage_annotate) == 1.0
    # ===End Guard===

    edge_list = dwug_sampling(trueGraph, {'simulationGraph': simulationGraph,
         'percentage_nodes': percentage_nodes, 'percentage_edges': percentage_edges,
         'min_size_mc': min_size_mc, 'num_flag': num_flag})

    start = 0
    for i, pa in enumerate(percentage_annotate):
        end = start + round(pa * len(edge_list)) if i < len(percentage_annotate) - 1 else len(edge_list)
        
        new_edge_list = []
        for j in range(start, end):
            new_weight = trueGraph.get_edge(*edge_list[j][:2], annotator=i, add_prob=0.5)
            new_edge_list.append((*edge_list[j][:2], new_weight))
        
        start = end
        simulationGraph.add_edges(new_edge_list, annotator=i)

    return None