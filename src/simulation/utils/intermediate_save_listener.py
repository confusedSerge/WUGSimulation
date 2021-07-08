import os
import pickle

from graphs.base_graph import BaseGraph
from simulation.runnable_step import RunnableStep


class IntermediateSaveListener(RunnableStep):
    """
    Listener for simulation, where for given checkpoints the graph is saved.
    """

    def __init__(self):
        self.checker = None
        self.function_to_listen = None

        self.checkpoints: list = None
        self.current_checkpoint = None
        self.next_checkpoint = None

        self.path: str = None
        self.id_prefix: str = None

    def add_listener(self, checkpoints: list, path: str, id_prefix: str, function_to_listen) -> None:
        self.checker = lambda cp, cc: cc <= cp
        self.function_to_listen = function_to_listen

        self.checkpoints = checkpoints
        self.current_checkpoint = checkpoints[0]
        self.next_checkpoint = 1

        self.path = path
        self._make_dir(self.path)
        self.id_prefix = id_prefix

    def add_fuzzy_listener(self, checkpoints: list, path: str, id_prefix: str, function_to_listen, fuzzyness: int = 5) -> None:
        self.checker = lambda cp, cc: cc <= cp or cc - fuzzyness <= cp
        self.function_to_listen = function_to_listen

        self.checkpoints = checkpoints
        self.current_checkpoint = checkpoints[0]
        self.next_checkpoint = 1

        self.path = path
        self._make_dir(self.path)
        self.id_prefix = id_prefix

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        assert self.checkpoints and self.path and self.id_prefix
        if len(self.checkpoints) <= self.next_checkpoint: return

        if not self.checker(self.function_to_listen(), self.current_checkpoint):
            return

        with open('{}/{}_{}.graph'.format(self.path, self.id_prefix, self.current_checkpoint), 'wb') as file:
            pickle.dump(annotated_graph, file)
        file.close()

        self.current_checkpoint = self.checkpoints[self.next_checkpoint]
        self.next_checkpoint += 1

    def clean_up(self):
        """
        Not needed
        """
        pass

    def _make_dir(self, path: str):
        try:
            os.makedirs('{}'.format(path))
        except FileExistsError:
            pass
