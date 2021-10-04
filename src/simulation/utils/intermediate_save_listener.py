import os
import pickle

from copy import deepcopy
from graphs.base_graph import BaseGraph
from simulation.runnable_step import RunnableStep
from visualization.graph_visualization import draw_graph_gt as draw


class IntermediateSaveListener(RunnableStep):
    """
    Listener for simulation, where for given checkpoints the graph is saved.
    """

    def __init__(self, tail_write: bool = False):
        self.checker = None
        self.function_to_listen = None

        self.checkpoints: list = []
        self.checkpoint_index = 0

        self.path: str = ''
        self.id_prefix: str = ''

        self.tail_write = tail_write
        self.graphs = []

        self.preprocessing_steps: list[RunnableStep] = []
        self.plot_save = False
        self.skip_oz = False

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

    def add_preprocessing_step(self, step: RunnableStep):
        self.preprocessing_steps.append(step)
        return self

    def save_draw(self):
        self.plot_save = True
        return self

    def skip_only_zeros(self):
        self.skip_oz = True
        return self

    def tail_write_function(self):
        for gaph, graph_path, draw_path in self.graphs:
            with open(graph_path, 'wb') as file:
                pickle.dump(gaph, file)
            file.close()

            draw(gaph, draw_path)

    def run(self, graph: BaseGraph, annotated_graph: BaseGraph) -> None:
        if self.skip_oz and len(annotated_graph.G.edges()) == 0:
            print('No edges added, skipping')
            return

        assert len(self.checkpoints) > 0 and self.path != '' and self.path is not None and self.id_prefix != '' and self.id_prefix is not None
        if not self.checkpoint_index < len(self.checkpoints):
            return

        if not self.checker(self.function_to_listen(), self.checkpoints[self.checkpoint_index]):
            return

        _annotated_graph = deepcopy(annotated_graph)

        if len(self.preprocessing_steps) > 0:
            for step in self.preprocessing_steps:
                step.run(graph, _annotated_graph)

        if self.tail_write:
            graph_path = '{}/{}{}.graph'.format(self.path, self.id_prefix, self.checkpoints[self.checkpoint_index])
            draw_path = '{}/{}{}.png'.format(self.path, self.id_prefix, self.checkpoints[self.checkpoint_index])
            self.graphs.append((_annotated_graph, graph_path, draw_path))

        else:
            with open('{}/{}{}.graph'.format(self.path, self.id_prefix, self.checkpoints[self.checkpoint_index]), 'wb') as file:
                pickle.dump(_annotated_graph, file)
            file.close()

            draw(_annotated_graph, '{}/{}{}.png'.format(self.path, self.id_prefix, self.checkpoints[self.checkpoint_index]))

        self.checkpoint_index += 1

    def clean_up(self):
        self.checkpoint_index = 0

    def _make_dir(self, path: str):
        try:
            os.makedirs('{}'.format(path))
        except FileExistsError:
            pass
