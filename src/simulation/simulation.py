from true_graph.true_graph import TrueGraph
from simulation.simulation_graph import SimulationGraph

"""
This module contains different simulations and can be extended to new ones.


"""

def simulation(tG: TrueGraph, sG: SimulationGraph = None, max_iter: int = 500, save_flag: bool = False, save_path: str = None, **params) -> (TrueGraph, SimulationGraph, bool):
    """
    Runs the simulation for the given iterations.
    Every iteration consists of a sampling phase, clustering (optional) phase, and checking the stopping criterion.

    Args:
        :param tG: True Graph used for sampling
        :param sG: Simulation graph, if provided continues the simulation based on this graph
        :param max_iter: maximal iterations to prevent non ending simulations, default 500 iterations
        :param save_flag: if results should be saved
        :param save_path: path to save to

        :param sampling_strategy: function to use for sampling
        :param sampling_params: params as dict for sampling function
        :param clustering_strategy: function to use for clustering (optional)
        :param clustering_params: params as dict for clustering function
        :param stopping_criterion: function to use as stopping criterion
        :param stopping_params: params as dict for stopping criterion
        :return tuple: returns a tuple containing the simulation graph, the true graph used, 
            and if maximal iteration was hit
    """
    if len(params) == 1:
        params = params['params']

    sampling_strategy = params.get('sampling_strategy', None)
    assert sampling_strategy != None
    
    sampling_params = params.get('sampling_params', None)
    assert type(sampling_params) == dict

    clustering_flag = False
    clustering_strategy = params.get('clustering_strategy', None)

    if clustering_strategy != None:
        clustering_flag = True

        clustering_params = params.get('clustering_params', None)
        assert type(clustering_params) == dict
    
    stopping_criterion = params.get('stopping_criterion', None)
    assert stopping_criterion != None
    
    stopping_params = params.get('stopping_params', None)
    assert type(stopping_params) == dict

    if sG == None:
        sG = SimulationGraph()

    for _ in range(max_iter):
        # sampling phase
        sampled_edges = sampling_strategy(tG, sampling_params)
        sG.add_edges(sampled_edges)

        # clustering phase
        if clustering_flag:
            clusters = clustering_strategy(sG, clustering_params)
            sG.update_community_membership(clusters)

        # stopping criterion
        if stopping_criterion(sG, stopping_params):
            break

    sG.update_graph_attributes()

    return (tG, sG, _ + 1 >= max_iter)


def simulation_with_tG_generator(tGs, max_iter: int = 500, save_flag: bool = False, save_path: str = None, **params) -> list:
    """
    See documentation simulation.
    It only differs in the True Graph input, as now a generator for tG is expected.
    """
    results = []
    for tG in tGs:
        results.append(simulation(tG, max_iter, save_flag, save_path, params=params))

    return results

