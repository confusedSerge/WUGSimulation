import os
import numpy as np

from copy import deepcopy

from simulation.runnable_step import RunnableStep
from graphs.base_graph import BaseGraph


class MetricListener(RunnableStep):
    """This Class calculates some metrics for given points and saves the results in a csv.
        It is comparable to the MetricRunner Class from the analysis module.
        Metric Methods should be of the same Signature as defined in the analysis module. 
    """

    def __init__(self, name: str, path: str, checkpoints: list, function_to_listen):
        """Init

        Args:
            name (str): Name of the file
            path (str): Path to save to
            checkpoints (list): At which points the metrics should be calculated
            function_to_listen (*function): which function to call to check if metric should be calculated
        """
        assert callable(function_to_listen)
        self.function_to_listen = function_to_listen

        self.checkpoints: list = checkpoints
        self.checkpoint_index = 0

        self.name: str = name
        self.path: str = path

        self.metric_file: str = '{}/{}.csv'.format(self.path, self.name)

        self._make_csv()

        self.metric_steps: list = []
        self.preprocessing_steps: list[RunnableStep] = []
        self.skip_oz = False

        self.graph_metrics = True

    def add_simple_metric(self, identifier: str, function, params: dict):
        assert callable(function) and type(params) == dict
        self.metric_steps.append(('simple', identifier, function, params))
        return self

    def add_comparison_metric(self, identifier: str, function, params: dict):
        assert callable(function) and type(params) == dict
        self.metric_steps.append(('comparison', identifier, function, params))
        return self

    def add_comparison_on_self_metric(self, identifier: str, function, params: dict):
        assert callable(function) and type(params) == dict
        self.metric_steps.append(('comparison_self', identifier, function, params))
        return self

    def add_preprocessing_step(self, step: RunnableStep):
        self.preprocessing_steps.append(step)
        return self

    def deactivate_graph_metrics(self):
        self.graph_metrics = False
        return self

    def skip_only_zeros(self):
        self.skip_oz = True
        return self

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        if self.skip_oz and len(annotated_graph.G.edges()) == 0:
            print('No edges added, skipping')
            return

        if not self.checkpoint_index < len(self.checkpoints) or len(self.metric_steps) == 0:
            return

        if not self.checkpoints[self.checkpoint_index] <= self.function_to_listen():
            return

        _annotated_graph = deepcopy(annotated_graph)

        if len(self.preprocessing_steps) > 0:
            for step in self.preprocessing_steps:
                step.run(graph, _annotated_graph)

        stats = [str(self.checkpoints[self.checkpoint_index])]
        for step in self.metric_steps:
            stats.append(str(self._calc(step, graph, _annotated_graph)))

        if self.graph_metrics:
            stats.extend(self._graph_metrics(_annotated_graph))

        if self.checkpoint_index == 0:
            self._add_header()

        with open(self.metric_file, 'a+') as file:
            file.write(',\t'.join(stats))
            file.write('\n')
        file.close()

        self.checkpoint_index += 1

    def _calc(self, step: tuple, graph: BaseGraph, annotated_graph: BaseGraph) -> float:
        if len(step) != 4 or not callable(step[2]) or type(step[3]) != dict:
            return 0.0
        try:
            if step[0] == 'simple':
                return step[2](annotated_graph, step[3])

            if step[0] == 'comparison_self':
                return step[2](annotated_graph, annotated_graph, step[3])

            if step[0] == 'comparison':
                return step[2](graph, annotated_graph, step[3])

        except Exception as identifier:
            print('Error with: {}'.format(identifier))
            return np.NAN

        return 0.0

    def _graph_metrics(self, graph: BaseGraph):
        graph_stats = []

        # add num edges
        graph_stats.append(len(graph.G.nodes()))

        # add num edges
        graph_stats.append(len(graph.G.edges()))

        # add num edges
        graph_stats.append(graph.get_num_added_edges())

        for ii, stats in enumerate(graph_stats):
            graph_stats[ii] = str(stats)

        return graph_stats

    def _make_csv(self):
        try:
            os.makedirs(self.path)
        except FileExistsError:
            pass

        try:
            open(self.metric_file, 'w').close()
        except FileExistsError:
            pass

    def _add_header(self):
        identifier = ['Checkpoint']
        for step in self.metric_steps:
            identifier.append(step[1])

        if self.graph_metrics:
            identifier.extend(['Nodes', 'Edges', 'Judgements'])

        with open(self.metric_file, 'a+') as file:
            file.write(',\t'.join(identifier))
            file.write('\n')
        file.close()
