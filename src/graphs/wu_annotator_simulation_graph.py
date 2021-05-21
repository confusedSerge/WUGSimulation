import numpy as np

from graphs.wu_simulation_graph import WUSimulationGraph

class WUAnnotatorSimulationGraph(WUSimulationGraph):

    def __init__(self, max_nodes, number_annotators):
        """
        Extension of WUG Simulation.

        Important: This is a specialized Graph which is used in special simulations
        """
        super().__init__(max_nodes)

        self.annotators = {}
        self.G.graph['edge_weight_hist'] = {}

        for i in range(number_annotators):
            self.annotators[i] = {'last_edge': None, 'judgements': 0}

    def add_edge(self, node_u: int, node_v: int, weight: float, annotator: int) -> None:
        """
        Extension of  WU Sim Graph
        Args:
            :param annotator: from which annotator did this annotation stem
        """
        self.add_edges([(node_u, node_v, weight)], annotator)

    def add_edges(self, edge_list: list, annotator: int = None) -> None:
        """
        Extension of  WU Sim Graph
        Args:
            :param annotator: from which annotator did this annotation stem
        """
        assert annotator == None or annotator in self.annotators.keys()

        if edge_list == None or len(edge_list) == 0:
            return

        median_edge_list = []

        for edge in edge_list:
            self.G.graph['edge_weight_hist'].get(edge[:2], []).append(edge[3])
            median_edge_list.append((*edge[:2], np.median(self.G.graph['edge_weight_hist'].get(edge[:2], []))))

        super().add_edges(median_edge_list)

        if annotator != None:
            self.annotators[annotator]['last_edge'] = median_edge_list[-1]
            self.annotators[annotator]['judgements'] = self.annotators[annotator]['judgements'] + len(median_edge_list)

    def get_last_added_edge(self, annotator: int = None):
        if annotator == None:
            return self.last_edge
        return self.annotators[annotator]['last_edge']

    def get_last_added_node(self, annotator: int = None):
        if annotator == None:
            return self.last_edge[1] if self.last_edge != None else None 
        return self.annotators[annotator]['last_edge'][1] if self.annotators[annotator]['last_edge'] != None else None

    def get_num_added_edges(self, annotator: int = None) -> int:
        if annotator == None:
            return self.judgements
        return self.annotators[annotator]['judgements']