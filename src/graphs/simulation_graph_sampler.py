import random
import numpy as np

from typing import List

from scipy.stats import lognorm

from graphs.utils.distribution import Binomial
from graphs.simulation_graph import SimulationGraph


class SimulationGraphSampler:

    def __init__(self, num_nodes, num_communities, size_communities, distribution):
        """
        Creates a simulation graph sampler, from witch new simulation graphs can be sampled.

        Args:
            num_nodes :          n, number of nodes. Can be int (static size) or tuple (rand choose)
            num_communities :    k, number of communities. Can be int (static size), tuple (rand choose), list (iterates through)
            size_communities :   c, size of each community, either (str, dict) (flag for dispensation of nodes across num_communities, **params) for calc, list (exact distribution)
            distribution :       v, list with first element describing which distribution is used and rest describing params
                example :   v = ['binomial', 3, 0.9]
                            v = ['binomial', (1, 3), (0.5, 0.9)]
                            v = ['binomial', 3, (0.5, 0.9)]
                            v = ['binomial', (1, 3), 0.9]
        """
        self.distr_flags = {'binomial': self._build_binomial_distr}
        self.dispensation_flags = {'random': self._random_dispensation_communities,
                                   'log': self._log_dispensation_communities,
                                   'log_iter': self._log_dispensation_communities_iter}

        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.size_communities = size_communities
        self.distribution_flag = distribution[0]
        self.distribution_data = distribution[1:]

    def sample_simulation_graph(self):
        """
        Samples a new simulation graph
        """
        community_dispensation, distribution = self._build_parameters()
        return SimulationGraph(community_dispensation, distribution=distribution)

    def _build_parameters(self):
        """
        Builds a new simulation graph
        """
        # ===Guard & Parameter Build Phase===
        # setup number nodes
        nodes = self.num_nodes
        assert type(nodes) == int or type(nodes) == tuple

        if type(nodes) == tuple:
            nodes = random.randint(*self.num_nodes)

        # setup number communities
        communities = self.num_communities
        assert type(communities) == int or type(
            communities) == tuple or type(communities) == list

        if type(self.num_communities) == list:
            raise AssertionError('num_communities cannot be a list')
        if type(self.num_communities) == tuple:
            communities = random.randint(*self.num_communities)

        # setup distribution used for edge weight
        if self.distribution_flag not in self.distr_flags:
            raise NotImplementedError("Distribution not implemented")
        distribution = self.distr_flags[self.distribution_flag](
            communities, *self.distribution_data)

        # setup node allocation to communities
        community_dispensation = self.size_communities
        if type(self.size_communities) == tuple:
            dispensation_flag, params = self.size_communities

            if dispensation_flag not in self.dispensation_flags:
                raise NotImplementedError("Dispensation not implemented")
            community_dispensation = self.dispensation_flags[dispensation_flag](
                nodes, communities, params)
        # ===Guard & Parameter Build Phase End===

        return community_dispensation, distribution

    @DeprecationWarning
    def sample_simulation_graph_generator(self):
        """
        Returns a simulation graph generator for the given k parameter.
        """

        nodes, rand_node_flag, distribution_method, communities, community_dispensation, community_dispensation_method_flag, community_dispensation_method, params = self._build_yield_parameters()

        for k in communities:
            # gen nodes
            if rand_node_flag:
                nodes = random.randint(*self.num_nodes)

            # gen community sizes
            if community_dispensation_method_flag:
                community_dispensation = community_dispensation_method(
                    nodes, k, params)

            # gen distribution
            distribution = distribution_method(k, *self.distribution_data)

            yield SimulationGraph(community_dispensation, distribution=distribution)

    def _build_yield_parameters(self):
        # ===Guard & Parameter Build Phase===
        # setup number nodes
        nodes = self.num_nodes
        assert type(nodes) == int or type(nodes) == tuple

        rand_node_flag = type(self.num_nodes) == tuple

        # check num_communities a list
        if type(self.num_communities) != list:
            raise AssertionError('num_communities not a list')
        communities = self.num_communities

        # check distributions existence & setup method
        if self.distribution_flag not in self.distr_flags:
            raise NotImplementedError("Distribution not implemented")
        distribution_method = self.distr_flags[self.distribution_flag]

        # setup community distribution
        community_dispensation = self.size_communities
        community_dispensation_method_flag = False
        if type(self.size_communities) == tuple:
            dist_flag, params = self.size_communities
            community_dispensation_method_flag = True

            if dist_flag not in self.dispensation_flags:
                raise NotImplementedError("Distribution not implemented")
            community_dispensation_method = self.dispensation_flags[dist_flag]
        # ===Guard & Parameter Build Phase===

        return nodes, rand_node_flag, distribution_method, communities, community_dispensation, community_dispensation_method_flag, community_dispensation_method, params
    # functions for building correct distribution

    def _build_binomial_distr(self, number_communities, tries, probability) -> Binomial:
        _tries = tries
        if (type(tries) == tuple):
            _tries = random.randint(*tries)

        _probability = probability
        if (type(probability) == tuple):
            _probability = random.randint(*probability)

        return Binomial(_tries, _probability, number_communities)

    # util functions

    """
    Functions to dispensate nodes across communities
    New dispensation can be added, but have to follow a specific method declaration
        def name(self, num_nodes: int, num_communities: int, params: dict) -> list:

    Also, new dispensation have to be added to the dispensation_flags
    """

    def _log_dispensation_communities(self, num_nodes: int, num_communities: int, params: dict) -> list:
        """
        Uses a lognorm probability density to calculate the sizes of each community

        Args:
            :params num_nodes: number of nodes to dispensate
            :params num_communities: number of communities (buckets) to dispensate to
            :params std_dev: standard deviation of the log_norm distribution
            :return list: dispensated nodes
        """
        assert type(params) == dict
        std_dev = params.get('std_dev', 0.5)

        community_split = lognorm.pdf(np.linspace(1, num_communities, num_communities), std_dev) * num_nodes
        community_split = [int(x) if int(x) > 0 else 1 for x in community_split]

        community_split[0] += num_nodes - sum(community_split)

        return community_split

    def _log_dispensation_communities_iter(self, num_nodes: int, num_communities: int, params: dict):
        """
        Uses a lognorm probability density to calculate the sizes of each community

        Args:
            :params num_nodes: number of nodes to dispensate
            :params num_communities: number of communities (buckets) to dispensate to
            :params std_dev: standard deviation of the log_norm distribution
            :params threshold: when rest nodes are below threshold, they will be added to the first bucket
            :return list: dispensated nodes
        """
        assert type(params) == dict
        std_dev = params.get('std_dev', 0.5)
        threshold = params.get('threshold', 5)

        r_nodes = num_nodes
        community_split_nodes = 0
        community_split_probability = lognorm.pdf(np.linspace(1, num_communities, num_communities), std_dev)

        while r_nodes > threshold:
            community_split_nodes += community_split_probability * r_nodes
            r_nodes = num_nodes - sum(community_split_nodes)

        community_split_nodes = np.array([int(x) if int(x) > 0 else 1 for x in community_split_nodes])
        community_split_nodes[0] += num_nodes - sum(community_split_nodes)
        return community_split_nodes

    def _random_dispensation_communities(self, num_nodes: int, num_communities: int, params: dict) -> list:
        """
        Return a randomly dispensated list of nodes.

        Args:
            :params num_nodes: number of nodes to dispensate
            :params num_communities: number of communities (buckets) to dispensate to
            :return list: dispensated nodes
        """

        dividers = sorted(random.sample(
            range(1, num_nodes), num_communities - 1))
        return [a - b for a, b in zip(dividers + [num_nodes], [0] + dividers)]
