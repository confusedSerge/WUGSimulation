import numpy as np

from analysis.metric_results import MetricResults
from graphs.base_graph import BaseGraph
from copy import deepcopy

class MetricRunner():
    """
    docstring
    """
    def __init__(self):
        self.steps = []
        self.metric_result = MetricResults()

    def add_simple_metric(self, function, params: dict) -> None:
        assert callable(function) and type(params) == dict 
        self.steps.append(('simple', function, params))

    def add_comparison_metric(self, function, params: dict) -> None:
        assert callable(function) and type(params) == dict 
        self.steps.append(('comparison', function, params))

    def run(self, graphs: list, reference_graph: BaseGraph) -> None:
        _mg = np.array(graphs)

        for step in self.steps:
            _graphs_metric = _mg.copy()
            it = np.nditer(np.zeros(_graphs_metric.shape), flags=['multi_index'])

            for _ in it:
                _graphs_metric[it.multi_index] = self._calc(step, _graphs_metric[it.multi_index], reference_graph)

            self.metric_result.add_metric_and_values(step[1].__name__, _graphs_metric)

    def _calc(self, step: tuple, graph: BaseGraph, ref_graph: BaseGraph) -> float:
        if len(step) != 3 or not callable(step[1]) or type(step[2]) != dict: return 0.0 
        if step[0] == 'simple':
            return step[1](graph, step[2])

        if step[0] == 'comparison':
            return step[1](ref_graph, graph, step[2])

    def reset(self):
        self.metric_result = MetricResults()

    def cleanup(self):
        self.steps = []
        self.metric_result = MetricResults()