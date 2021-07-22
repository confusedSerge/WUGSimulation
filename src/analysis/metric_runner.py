import numpy as np

from analysis.metric_results import MetricResults
from graphs.base_graph import BaseGraph
from copy import deepcopy

class MetricRunner():
    """
    docstring
    """
    def __init__(self, name, info):
        self.steps = []
        self.metric_result = MetricResults(name=name, info=info)

    def add_simple_metric(self, function, params: dict, identifier: str = None):
        assert callable(function) and type(params) == dict 
        self.steps.append(('simple', function, params, identifier))
        return self

    def add_comparison_metric(self, function, params: dict, identifier: str = None):
        assert callable(function) and type(params) == dict 
        self.steps.append(('comparison', function, params, identifier))
        return self

    def add_comparison_on_self_metric(self, function, params: dict, identifier: str = None):
        assert callable(function) and type(params) == dict 
        self.steps.append(('comparison_self', function, params, identifier))
        return self

    def run(self, graphs: list, reference_graphs: list) -> None:
        _mg = np.array(graphs)
        _rf = np.array(reference_graphs)
        assert _mg.shape == _rf.shape

        for step in self.steps:
            _graphs_metric = _mg.copy()
            _ref_graphs = _rf.copy()
            it = np.nditer(np.zeros(_graphs_metric.shape), flags=['multi_index'])

            for _ in it:
                _graphs_metric[it.multi_index] = self._calc(step, _graphs_metric[it.multi_index], _ref_graphs[it.multi_index])

            self.metric_result.add_metric_and_values(step[1].__name__ if step[3] == None else step[3], _graphs_metric)

    def _calc(self, step: tuple, graph: BaseGraph, ref_graph: BaseGraph) -> float:
        if len(step) != 4 or not callable(step[1]) or type(step[2]) != dict: return 0.0 
        if step[0] == 'simple':
            return step[1](graph, step[2])

        if step[0] == 'comparison_self':
            return step[1](graph, graph, step[2])

        if step[0] == 'comparison':
            return step[1](ref_graph, graph, step[2])

        return 0.0

    def reset(self):
        self.metric_result = MetricResults()

    def cleanup(self):
        self.steps = []
        self.metric_result = MetricResults()