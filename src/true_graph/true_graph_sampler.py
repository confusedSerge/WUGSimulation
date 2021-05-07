import random
import numpy as np

from scipy.stats import lognorm

from true_graph.distribution import *
from true_graph.true_graph import TrueGraph

class TrueGraphSampler:

    def __init__(self, num_nodes, num_communities, size_communities, distribution):
        """

        Args:
            num_nodes :          n, number of nodes. Can be int (static size) or tuple (rand choose)
            num_communities :    k, number of communities. Can be int (static size), tuple (rand choose), list (iterates through)
            size_communities :   c, size of each community, either (str, params_list) (flag for distributing nodes across num_communities, param) for calc, list (exact distribution)
            distribution :       v, list with first element describing which distribution is used and rest describing params
                example :   v = ['binomial', 3, 0.9]
                            v = ['binomial', (1, 3), (0.5, 0.9)]
                            v = ['binomial', 3, (0.5, 0.9)]
                            v = ['binomial', (1, 3), 0.9]
        """
        self.distr_flags = {'binomial': self._build_binomila_distr}
        self.com_flags = {'random': self._random_distribute_communities,
                          'log': self._log_distribute_communities}

        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.size_communities = size_communities
        self.distribution_flag = distribution[0]
        self.distribution_data = distribution[1:]

    def sample_graph(self):
        # setup number nodes
        nodes = self.num_nodes
        if type(nodes) == tuple:
            nodes = random.randint(*self.num_nodes)

        # setup number communities
        communities = self.num_communities
        if type(self.num_communities) == list:
            raise AssertionError('num_communities cannot be a list')
        if type(self.num_communities) == tuple:
            communities = random.randint(*self.num_communities)

        # setup distribution used for edge weight
        if self.distribution_flag not in self.distr_flags:
            raise NotImplementedError("Distribution not implemented")
        distribution = self.distr_flags[self.distribution_flag](communities, *self.distribution_data)

        # setup node allocation to communities
        community_dispensation = self.size_communities
        if type(self.size_communities) == tuple:
            dist_flag, params = self.size_communities

            if dist_flag not in self.com_flags:
                raise NotImplementedError("Distribution not implemented")
            community_dispensation = self.com_flags[dist_flag](nodes, communities, params)

        return TrueGraph(community_dispensation, distribution=distribution)

    def sample_graph_generator(self):
        # setup number nodes
        nodes = self.num_nodes
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

            if dist_flag not in self.com_flags:
                raise NotImplementedError("Distribution not implemented")
            community_dispensation_method = self.com_flags[dist_flag]

        for k in communities:
            # gen nodes
            if rand_node_flag:
                nodes = random.randint(*self.num_nodes)

            # gen community sizes
            if community_dispensation_method_flag:
                community_dispensation = community_dispensation_method(nodes, k, params)

            # gen distribution
            distribution = distribution_method(k, *self.distribution_data)

            yield TrueGraph(community_dispensation, distribution=distribution)

    # functions for building correct distribution

    def _build_binomila_distr(self, number_communities, tries, probability):
        _tries = tries
        if (type(tries) == tuple):
            _tries = random.randint(*tries)

        _probability = probability
        if (type(probability) == tuple):
            _probability = random.randint(*probability)

        return Binomial(_tries, _probability, number_communities)

    # functions to distibute nodes across communities
    # TODO Fix that params solution, its not nice
    def _log_distribute_communities(self, num_nodes, num_communities, params):
        """
        TODO: Fix Problem with std_dev close to 1 adding too many edges to first community 

        Uses a lognorm probability density to calculate the sizes of each community
        """
        if (len(params) == 0):
            std_dev = 0.5
        else:
            std_dev = params[0]

        community_split = lognorm.pdf(np.linspace(
            1, num_communities, num_communities), std_dev) * num_nodes
        community_split = [int(x) if int(
            x) > 0 else 1 for x in community_split]

        community_split[0] += num_nodes - sum(community_split)

        return community_split

    def _random_distribute_communities(self, num_nodes, num_communities, params):
        """
        Return a randomly chosen list of n positive integers summing to total.
        Each such list is equally likely to occur.
        Params Arg is just for simplicity and does nothing!
        """

        dividers = sorted(random.sample(
            range(1, num_nodes), num_communities - 1))
        return [a - b for a, b in zip(dividers + [num_nodes], [0] + dividers)]
