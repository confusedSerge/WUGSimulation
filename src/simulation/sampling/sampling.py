import numpy as np

from graphs.base_graph import BaseGraph
from simulation.runnable_step import RunnableStep
from simulation.sampling.annotator import Annotator


class Sampling(RunnableStep):
    """
    Note, using 'per-annotator' will perform the sampling per annotator, 
        which could result in more sampled edges as expected, if the values are not adjusted
    """

    def __init__(self):
        super().__init__()
        self.annotators: list = []
        self.annotator_dist: str = 'none'

        # none: sampling will be done without annotators,
        # per: sampling will be done per annotator,
        # across: each annotator will get a non-overlapping subset of the edge-list
        # random: annotators will be chosen randomly
        self.current_annotator_dist = {'none': self._run_normal_sampling,
                                       'per': self._run_per_annotator,
                                       'across': self._run_across_annotators,
                                       'random': self._run_random_annotators}

    def add_sampling_strategie(self, function, params: dict) -> None:
        """
        Add Sampling Strategies
        """
        self.complexity = 'simple'
        self.function = function
        self.params = params

    def add_adv_sampling_strategie(self, function, params: dict, clean_up_func) -> None:
        """
        Add Sampling Strategies from the advanced module
        """
        self.complexity = 'adv'
        self.function = function
        self.params = params
        self.clean_up_func = clean_up_func

    def add_annotator(self, annotator: Annotator) -> None:
        self.annotators.append(annotator)

    def set_annotator_dist(self, annotator_dist) -> None:
        assert annotator_dist in self.current_annotator_dist
        self.annotator_dist = annotator_dist

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        """
        Rung given sampling
        """
        assert self.complexity and self.function and self.params
        self.current_annotator_dist[self.annotator_dist](
            graph, annotated_graph)

    def _run_normal_sampling(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        annotated_graph.add_edges(
            self._sample_edge_list(graph, annotated_graph))

    def _run_across_annotators(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        assert len(self.annotators) > 0

        edge_list = self._sample_edge_list(graph, annotated_graph)

        n = int(len(edge_list) / len(self.annotators))
        r = len(edge_list) % len(self.annotators)

        for i, annotator in enumerate(self.annotators):
            for j in range(n):
                edge_list[j + i * n] = (*edge_list[j + i * n][:2],
                                        annotator.error_prone_sampling(edge_list[j + i * n][2]))

        for i in range(r):
            edge_list[i - r] = (*edge_list[i - r][:2],
                        annotator.error_prone_sampling(edge_list[i - r]))

        annotated_graph.add_edges(
            self._sample_edge_list(graph, annotated_graph))

    def _run_random_annotators(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        assert len(self.annotators) > 0

        edge_list = self._sample_edge_list(graph, annotated_graph)

        for j in range(len(edge_list)):
            annotator = np.random.choice(self.annotators)
            edge_list[j] = (*edge_list[j][:2],
                            annotator.error_prone_sampling(edge_list[j][2]))

        annotated_graph.add_edges(edge_list)

    def _run_per_annotator(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        assert len(self.annotators) > 0

        for annotator in self.annotators:
            edge_list = self._sample_edge_list(graph, annotated_graph)
            for j in range(len(edge_list)):
                edge_list[j] = (*edge_list[j][:2],
                                annotator.error_prone_sampling(edge_list[j][2]))
            annotated_graph.add_edges(edge_list)

    def _sample_edge_list(self, graph: BaseGraph, annotated_graph: BaseGraph) -> list:
        if self.complexity == 'simple':
            edge_list = self.function(graph, self.params)
        if self.complexity == 'adv':
            edge_list = self.function(graph, annotated_graph, self.params)

        return edge_list

    def clean_up(self):
        """
        Cleanup of sampling
        """
        if callable(self.sampling_clean_up_func):
            self.sampling_clean_up_func()
