from graphs.base_graph import BaseGraph
from simulation.runnable_step import RunnableStep

class Simulation(RunnableStep):
    """
    # TODO: Maybe flag handling through return
    """

    def __init__(self, max_iter: int = 5000, break_on_sc: bool = True):
        self.steps: list = []
        self.max_iter: int = max_iter
        
        # Currently does nothing
        self.break_on_sc: bool = break_on_sc

    def add_step(self, step: RunnableStep):
        self.steps.append(step)
        return self

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        for _ in range(self.max_iter):
            for step in self.steps:
                step.run(graph, annotated_graph)

    def clean_up(self):
        for step in self.steps:
            step.clean_up()
