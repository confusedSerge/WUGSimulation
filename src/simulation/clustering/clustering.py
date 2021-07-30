from graphs.base_graph import BaseGraph
from simulation.runnable_step import RunnableStep


class Clustering(RunnableStep):
    """
    docstring
    """

    def __init__(self):
        super().__init__()

    def add_clustering_strategy(self, function, params: dict):
        """
        Add Clustering Strategies
        """
        self.complexity = 'simple'
        self.function = function
        self.params = params
        return self

    def add_adv_clustering_strategy(self, function, params: dict, clean_up_func):
        """
        Add Clustering Strategies from the advanced module
        """
        self.complexity = 'adv'
        self.function = function
        self.params = params
        self.clean_up_func = clean_up_func
        return self

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        """
        Run clustering
        """
        assert callable(self.function) and self.params is not None

        if self.complexity == 'simple' or self.complexity == 'adv':
            clusters = self.function(annotated_graph, self.params)

        annotated_graph.update_community_nodes_membership(clusters)

    def clean_up(self) -> None:
        """
        Cleanup of clustering
        """
        if callable(self.clean_up_func):
            self.clean_up_func()
