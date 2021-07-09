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

        self.checkpoints: list = []
        self.checkpoint_index = 0

        self.path: str = ''
        self.id_prefix: str = ''

    def add_listener(self, checkpoints: list, path: str, id_prefix: str, function_to_listen):
        self.checker = lambda cp, cc: cc <= cp
        self.function_to_listen = function_to_listen

        self.checkpoints = checkpoints

        self.path = path
        self._make_dir(self.path)
        self.id_prefix = id_prefix

        return self

    def add_fuzzy_listener(self, checkpoints: list, path: str, id_prefix: str, function_to_listen, fuzzyness: int = 5):
        self.checker = lambda cp, cc: cc <= cp or cc - fuzzyness <= cp
        self.function_to_listen = function_to_listen

        self.checkpoints = checkpoints

        self.path = path
        self._make_dir(self.path)
        self.id_prefix = id_prefix

        return self

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        assert len(self.checkpoints) > 0 and self.path != '' and self.path != None and self.id_prefix != '' and self.id_prefix != None
        if not self.checkpoint_index < len(self.checkpoints): return

        if not self.checker(self.function_to_listen(), self.checkpoints[self.checkpoint_index]):
            return

        with open('{}/{}_{}.graph'.format(self.path, self.id_prefix, self.checkpoints[self.checkpoint_index]), 'wb') as file:
            pickle.dump(annotated_graph, file)
        file.close()

        self.checkpoint_index += 1

    def clean_up(self):
        self.checkpoint_index = 0

    def _make_dir(self, path: str):
        try:
            os.makedirs('{}'.format(path))
        except FileExistsError:
            pass
