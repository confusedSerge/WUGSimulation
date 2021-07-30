from graphs.base_graph import BaseGraph


class RunnableStep():
    """
    docstring
    """

    def __init__(self):
        self.complexity: str = None
        self.function = None
        self.params: dict = None
        self.clean_up_func = None

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        raise NotImplementedError

    def clean_up(self):
        raise NotImplementedError
