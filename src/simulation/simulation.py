import time

from graphs.base_graph import BaseGraph
from simulation.runnable_step import RunnableStep


class Simulation(RunnableStep):
    """
    # TODO: Maybe flag handling through return
    """

    def __init__(self, max_iter: int = 5000, break_on_sc: bool = True, tail_write: bool = False, verbose: bool = False):
        self.steps: list = []
        self.max_iter: int = max_iter

        self.verbose = verbose
        self.break_on_sc: bool = break_on_sc
        self.tail_write = tail_write

    def add_step(self, step: RunnableStep):
        self.steps.append(step)
        return self

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        break_called = False
        for _ in range(self.max_iter):
            if self.verbose:
                round_start_time = time.time()
                print('Round: ', _)
            for step in self.steps:
                if self.verbose:
                    print('Current Step: ', step.__class__.__name__)
                    step_start_time = time.time()

                step.run(graph, annotated_graph)

                if self.verbose:
                    step_execution_time = (time.time() - step_start_time)
                    print('Step-Execution time in seconds: ' + str(step_execution_time))

                if self.break_on_sc and step.get_break():
                    # TODO: Make so all steps?
                    print('Step called Break')
                    break_called = True

            if self.verbose:
                round_execution_time = (time.time() - round_start_time)
                print('Round-Execution time in seconds: ' + str(round_execution_time), '\n')

            if break_called:
                break

        if self.tail_write:
            round_start_time = time.time()

            for step in self.steps:
                step.tail_write_function()

            round_execution_time = (time.time() - round_start_time)
            print('Tail-Write-Execution time in seconds: ' + str(round_execution_time))

    def clean_up(self):
        for step in self.steps:
            step.clean_up()
