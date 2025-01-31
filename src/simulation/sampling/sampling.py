import numpy as np

from graphs.base_graph import BaseGraph
from simulation.runnable_step import RunnableStep
from simulation.sampling.annotator import Annotator


class Sampling(RunnableStep):
    """
    Note, using 'per-annotator' will perform the sampling per annotator,
        which could result in more sampled edges as expected, if the values are not adjusted
    """

    def __init__(self, annotations_per_edge: int = 1):
        '''Init Sampling step.

        Parameters
        ----------

        annotations_per_edge: number of annotations_per_edge.
            Currently only works for random annotators
        '''
        super().__init__()
        self.annotators: list = []
        self.annotator_dist: str = 'none'
        self.clean_up_func = None

        # How often an edge will be annotated by annotators
        self.annotations_per_edge = annotations_per_edge

        # none: sampling will be done without annotators,
        # per: sampling will be done per annotator,
        # across: each annotator will get a non-overlapping subset of the edge-list
        # random: annotators will be chosen randomly
        self.current_annotator_dist = {'none': self._run_normal_sampling,
                                       'per': self._run_per_annotator,
                                       'across': self._run_across_annotators,
                                       'random': self._run_random_annotators}

    def add_sampling_strategie(self, function, params: dict):
        """
        Add Sampling Strategies
        """
        self.complexity = 'simple'
        self.function = function
        self.params = params
        return self

    def add_adv_sampling_strategie(self, function, params: dict, clean_up_func):
        """
        Add Sampling Strategies from the advanced module
        """
        self.complexity = 'adv'
        self.function = function
        self.params = params
        self.clean_up_func = clean_up_func
        return self

    def add_annotator(self, annotator: Annotator):
        self.annotators.append(annotator)
        return self

    def set_annotator_dist(self, annotator_dist):
        assert annotator_dist in self.current_annotator_dist
        self.annotator_dist = annotator_dist
        return self

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        """
        Run given sampling
        """
        assert callable(self.function) and self.params is not None
        self.current_annotator_dist[self.annotator_dist](
            graph, annotated_graph)

    def _run_normal_sampling(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        annotated_graph.add_edges(
            self._sample_edge_list(graph, annotated_graph))

    def _run_across_annotators(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        assert len(self.annotators) > 0

        edge_list = self._sample_edge_list(graph, annotated_graph)

        if len(edge_list) == 0:
            return

        n = int(len(edge_list) / len(self.annotators))
        r = len(edge_list) % len(self.annotators)

        for i, annotator in enumerate(self.annotators):
            for j in range(n):
                edge_list[j + i * n] = (*edge_list[j + i * n][:2],
                                        annotator.error_prone_sampling(*edge_list[j + i * n]))

        for i in range(r):
            edge_list[i - r] = (*edge_list[i - r][:2],
                                self.annotators[-1].error_prone_sampling(*edge_list[i - r]))

        annotated_graph.add_edges(
            self._sample_edge_list(graph, annotated_graph))

    def _run_random_annotators(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        assert len(self.annotators) > 0

        edge_list = self._sample_edge_list(graph, annotated_graph)

        if len(edge_list) == 0:
            return

        annotated_edge_list = []
        for j in range(len(edge_list)):
            for _ in range(self.annotations_per_edge):
                annotator = np.random.choice(self.annotators)
                annotated_edge_list.append((*edge_list[j][:2], annotator.error_prone_sampling(*edge_list[j])))

        annotated_graph.add_edges(annotated_edge_list)

    def _run_per_annotator(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        assert len(self.annotators) > 0

        for annotator in self.annotators:
            edge_list = self._sample_edge_list(graph, annotated_graph)
            if len(edge_list) == 0:
                continue

            for j in range(len(edge_list)):
                edge_list[j] = (*edge_list[j][:2],
                                annotator.error_prone_sampling(*edge_list[j]))
            annotated_graph.add_edges(edge_list)

    def _sample_edge_list(self, graph: BaseGraph, annotated_graph: BaseGraph) -> list:
        if self.complexity == 'simple':
            edge_list = self.function(graph, self.params)
        elif self.complexity == 'adv':
            edge_list = self.function(graph, annotated_graph, self.params)
        else:
            edge_list = []

        return edge_list

    def clean_up(self):
        """
        Cleanup of sampling
        """
        if callable(self.clean_up_func):
            self.clean_up_func()
