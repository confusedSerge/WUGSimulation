import pickle
import numpy as np
import random

from graspologic.simulations import sbm


# TODO typing

class Distribution:

    def __init__(self):
        pass

    def get_distribution(self):
        raise NotImplementedError

    def get_dist_param(self):
        raise NotImplementedError

    def get_dist_param_dict(self) -> dict:
        raise NotImplementedError


class Binomial(Distribution):

    def __init__(self, tries: int, probability: float, size: int):
        self.distribution = np.random.binomial
        self.tries = tries
        self.probability = probability
        self.probability_dict = {}

        self._prob_dict_builder(size)

    def _prob_dict_builder(self, size: int) -> []:
        true_prob = [dict(n=self.tries, p=self.probability)]
        inv_prob = [dict(n=self.tries, p=1 - self.probability)]*(size - 1)

        true_prob.extend(inv_prob)
        prob_dict = [true_prob]

        for i in range(1, size):
            prob_dict.append(true_prob[-i:] + true_prob[:-i])

        self.probability_dict = prob_dict

    def get_distribution(self):
        return self.distribution

    def get_dist_param(self):
        return self.get_dist_param

    def get_dist_param_dict(self) -> dict:
        return self.probability_dict


class GraphSimulator:

    # TODO should this be reworked?

    def __init__(self):
        self.G = None
        self.labels = None

    def gen_graph_from_params(self, communities: list, communities_probability: list = None, distribution: Distribution = None):
        """
        Generates a graph from given parameters.
        If communities_probability is not given, it will generate a complete graph
        TODO
        Args:
            TODO
        """

        if distribution == None:
            raise AssertionError

        if communities_probability == None:
            communities_probability = np.ones(
                (len(communities), len(communities)), np.int)

        self.G, self.labels = sbm(n=communities, p=communities_probability, wt=distribution.get_distribution(
        ), wtargs=distribution.get_dist_param_dict(), return_labels=True)

        self.G += 1

    def sample_edge(self, u_node: int, v_node:int) -> int:
        return int(self.G[u_node][v_node]) 

    def save_graph(self, path: str):
        with open(path, "xb") as file:
            pickle.dump(self, file)
        file.close()

    @staticmethod
    def load_graph(path: str):
        with open(path, "rb") as file:
            graph = pickle.load(file)
        file.close()
        return graph

    def __str__(self):
        return 'Adjacency Matrix {}, Labels {}'.format(self.G, self.labels)

class GraphSimulatorSamplerFactory:

    # TODO add guards
    # TODO add communiti prob
    def __init__(self):
        self.node_data = ()
        self.communities_data = ()
        self.distribution_flag = None
        self.distribution_data = ()

    def add_ul_bound_nodes(self, lower_bound: int, uper_bound: int):
        self.node_data = (lower_bound, uper_bound)
        return self

    def add_ul_bound_communities(self, lower_bound: int, uper_bound: int):
        self.communities_data = (lower_bound, uper_bound)
        return self

    def add_dist_binomial(self, lower_bound_tries: int, uper_bound_tries: int, lower_bound_prob: float, uper_bound_prob: float):
        self.distribution_flag = 'binomial'
        self.distribution_data = ((lower_bound_tries, uper_bound_tries), (lower_bound_prob, uper_bound_prob))
        return self

    def build_graph_simulator_sampler(self):
        return GraphSimulatorSampler(self.node_data, self.communities_data, self.distribution_flag, self.distribution_data)

class GraphSimulatorSampler:

    # TODO add communiti prob
    def __init__(self, node_data, communities_data, distr_flag: str, distribution_data):
        self.distr_flags = {'binomial': self._build_binomila_distr}
        self.dist_builder = None

        if distr_flag in self.distr_flags:
            self.dist_builder = self.distr_flags[distr_flag]
        else:
            raise NotImplementedError("Distribution not implemented")

        self.node_data = node_data
        self.communities_data = communities_data
        self.distribution_data = distribution_data

    def sample_graph(self, add_info_flag: bool = False):
        number_nodes = random.randint(*self.node_data)
        number_communities = random.randint(*self.communities_data)

        node_com = self._divide_nodes_into_communities(number_communities, number_nodes)

        distribution = self.dist_builder(self.distribution_data, number_communities)

        graph = GraphSimulator()
        graph.gen_graph_from_params(node_com, distribution=distribution)

        if add_info_flag:
            return graph, (number_nodes, number_communities, node_com, distribution)
        return graph

    def _build_binomila_distr(self, data, size_community):
        lu_tries, lu_prob = data
        number_tries = random.randint(*lu_tries)
        probability = random.uniform(*lu_prob)

        return Binomial(number_tries, probability, size_community)
        

    def _divide_nodes_into_communities(self, number_communities, number_nodes):
        """Return a randomly chosen list of n positive integers summing to total.
        Each such list is equally likely to occur."""

        dividers = sorted(random.sample(range(1, number_nodes), number_communities - 1))
        return [a - b for a, b in zip(dividers + [number_nodes], [0] + dividers)]