import pickle
import numpy as np
import networkx as nx

from graphs.wu_graph import WUGraph
from graphs.utils.annotator import Annotator

class WUAnnotatorGraph(WUGraph):

    def __init__(self, communities, communities_probability=None, distribution=None, annotators: list = []):
        """
        Extension of WUG implementation

        Important: This is a specialized Graph which is used in special simulations
        """
        super().__init__(communities, communities_probability=communities_probability, distribution=distribution)

        self.annotators = annotators

    def get_edge(self, u_node: int, v_node: int, **params) -> int:
        """
        Same as WUGraphs get_edge function, but if annotators are given and specified,
            returns the edgeweight of the graph with an error term.

        Additional Args:
            :param annotator: int describing which annotator
            :param add_prob: probabilty describing if error added or subtracted
            :returns: edgeweight with/out error
        """
        annotator = params.get('annotator', None)
        assert type(annotator) == int or annotator == None

        if len(self.annotators) == 0 or annotator == None:
            return super().get_edge(u_node, v_node)

        p = params.get('add_prob', 1)
        assert 0 <= p <= 1

        error = self.annotators[annotator].sample_error()
        coefficient = np.random.choice([1, -1], p=[p, 1 - p])
        
        return np.min([np.max([super().get_edge(u_node, v_node) + coefficient * error, 1]), 4])

    def get_num_annotators(self):
        return len(self.annotators)