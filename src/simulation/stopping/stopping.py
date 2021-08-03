from copy import deepcopy

from graphs.base_graph import BaseGraph
from simulation.runnable_step import RunnableStep


class Stopping(RunnableStep):
    """
    docstring
    """

    def __init__(self):
        super().__init__()
        self.sc_hit_flag: bool = False
        self.sc_annotated_graph: BaseGraph = None

    def add_stopping_criterion(self, function, params: dict):
        """
        Add Stopping Strategies
        """
        self.complexity = 'simple'
        self.function = function
        self.params = params
        return self

    def add_adv_stopping_criterion(self, function, params: dict, clean_up_func):
        """
        Add Stopping Strategies from the advanced module
        """
        self.complexity = 'adv'
        self.function = function
        self.params = params
        self.clean_up_func = clean_up_func
        return self

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        """
        Run Stopping criterion
        """
        assert callable(self.function) and self.params is not None

        if self.complexity == 'simple' or self.complexity == 'adv':
            sc_flag = self.function(annotated_graph, self.params)

        if not self.sc_hit_flag and sc_flag:
            self.sc_annotated_graph = deepcopy(annotated_graph)
            self.sc_hit_flag = True

    def clean_up(self) -> None:
        """
        Cleanup
        """
        self.sc_hit_flag: bool = False
        self.sc_annotated_graph: BaseGraph = None
        if callable(self.clean_up_func):
            self.clean_up_func()
