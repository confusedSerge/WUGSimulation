
from graphs.base_graph import BaseGraph

"""
This module adds an analyzer, which can have multiple metrics as input and returns their result in a dict
"""

def analyze(trueGraph: BaseGraph, simulatedGraph: BaseGraph, **params) -> dict:
    """
    This function takes the two graphs that will be analyzed, based on the metrics and parameters given.
    Expected input:
        analyze(tG, sG, metric1=(funct_pointer, {parameters}), metric2=(funct_pointer, {parameters}), ...)
    Output:
        dict = {"metric1":(result), "metric2":(result), ...}
    Args:
        :param tG: True Graph
        :param sG: predicted/simulated graph
        :param **params: metrics to use
        :res: predicted/simulated graph
    """

    result = {}

    for k, v in params.items():
        result[k] = v[0](trueGraph, simulatedGraph, v[1])

    return result
