from graphs.base_graph import BaseGraph
from copy import deepcopy

def full_simulation(trueGraph: BaseGraph, simulationGraph: BaseGraph, max_iter: int = 50000, verbose: bool = False, **params) -> (BaseGraph, bool, list, list):
    """
    Runs a full simulation for the given parameters, (with a limit).
    Every iteration consists of a sampling phase, clustering (optional) phase, analyzing phase (based on some points), and checking the stopping criterion.

    Args:
        :param tG: True Graph used for sampling
        :param simulationGraph: Simulation graph, if provided starts/continues the simulation based on this graph
        :param max_iter: maximal iterations to prevent non ending simulations, default 500 iterations
        :param verbose: flag

        :param sampling_strategy: function to use for sampling
        :param sampling_params: params as dict for sampling function

        :param clustering_strategy: function to use for clustering (optional)
        :param clustering_params: params as dict for clustering function

        :param stopping_criterion: function to use as stopping criterion
        :param stopping_params: params as dict for stopping criterion

        :param analyzing_critertion: function to determine when to analyze 
        :param analyzing_critertion_params: params as list of dict for analyzing criterion, for each point

        :param anal_clustering_strategy: function to use for clustering befor analyzing
        :param anal_clustering_params: params as dict for clustering function

        :param analyzing_func: function used for analyzing 
        :param analyzing_params: params as dict for analyzing function
        :param return_graph_flag: if analyzed graph should be returned

        :return tuple: returns a tuple containing the simulation graph, if maximal iteration was hit, list of metrics collected at diff points and corresponding graph
    """
    if verbose: print('Started Guard-Phase')

    assert isinstance(trueGraph, BaseGraph)
    assert isinstance(simulationGraph, BaseGraph)

    if len(params) == 1:
        params = params['params']

    # Get Sampling
    sampling_strategy = params.get('sampling_strategy', None)
    assert sampling_strategy != None
    
    sampling_params = params.get('sampling_params', None)
    assert type(sampling_params) == dict

    # Get Clustering
    clustering_flag = False
    clustering_strategy = params.get('clustering_strategy', None)

    if clustering_strategy != None:
        clustering_flag = True

        clustering_params = params.get('clustering_params', None)
        assert type(clustering_params) == dict
    
    # Get Stopping
    stopping_criterion = params.get('stopping_criterion', None)
    assert stopping_criterion != None
    
    stopping_params = params.get('stopping_params', None)
    assert type(stopping_params) == dict

    # Get Analyzing criterion
    analyzing_critertion = params.get('analyzing_critertion', None)
    assert analyzing_critertion != None
    
    analyzing_critertion_params = params.get('analyzing_critertion_params', None)
    assert type(analyzing_critertion_params) == list

    current_acp = analyzing_critertion_params[0]
    current_acp_counter = 0
    acp_list = []

    # Get Analyzing Clustering
    anal_clustering_strategy = params.get('anal_clustering_strategy', None)
    assert anal_clustering_strategy != None
    
    anal_clustering_params = params.get('anal_clustering_params', None)
    assert type(anal_clustering_params) == dict

    # Get Analyzing 
    analyzing_func = params.get('analyzing_func', None)
    assert analyzing_func != None
    
    analyzing_params = params.get('analyzing_params', None)
    assert type(analyzing_params) == dict

    return_graph_flag = params.get('return_graph_flag', None)
    assert type(return_graph_flag) == bool

    return_graph = []

    if verbose: print('Started Sim-Phase')
    # simulation
    for _ in range(max_iter):
        # sampling phase
        sampled_edges = sampling_strategy(trueGraph, sampling_params)
        simulationGraph.add_edges(sampled_edges)

        # clustering phase
        if clustering_flag:
            clusters = clustering_strategy(simulationGraph, clustering_params)
            simulationGraph.update_community_nodes_membership(clusters)

        # analyzing phase
        if current_acp_counter != None and analyzing_critertion(simulationGraph, current_acp):
            # clustering
            clusters = anal_clustering_strategy(simulationGraph, anal_clustering_params)
            simulationGraph.update_community_nodes_membership(clusters)
            # analyzing
            tmp = analyzing_func(trueGraph, simulationGraph, params=analyzing_params)
            acp_list.append(tmp)
            # save graph
            if return_graph_flag:
                return_graph.append(deepcopy(simulationGraph))

            # get next acp
            current_acp_counter += 1
            if current_acp_counter < len(analyzing_critertion_params):
                current_acp = analyzing_critertion_params[current_acp_counter]
            else:
                current_acp_counter = None

        # stopping criterion
        if stopping_criterion(simulationGraph, stopping_params):
            break
        
    if verbose: print('Finished')
    return (simulationGraph, _ + 1 >= max_iter, acp_list, return_graph)