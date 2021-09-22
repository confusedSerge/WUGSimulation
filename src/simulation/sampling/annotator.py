import numpy as np


class Annotator():

    def __init__(self):
        r""" Annotator Class, shich simulates certain effects of annotator.
        Includes:
            - Annotation error for a sample
            - High error nodes
            - Zero annotation (currently only uniformly chosen based on probability)

        We expect sth like:
            annotator = Annotator()
            annotator.add_error_sampling(np.random.poisson, dict(lam=1), 1, 4, 0.5)
                     .add_zero_probability(0.01)
        """
        # Parameters for normal error prone sampling
        self.distribution = None
        self.param = None
        self.minimum = None
        self.maximum = None
        self.add_probability = None

        # Nodes with high error probability
        self.high_error_nodes = []
        self.he_distribution = None
        self.he_param = None
        self.he_minimum = None
        self.he_max = None
        self.he_add_prob = None

        # Probability for zero sample
        self.zero_probability = 0

    def add_error_sampling(self, distribution, param: dict, minimum: float, maximum: float, add_probability: float):
        self.distribution = distribution
        self.param = param
        self.minimum = minimum
        self.maximum = maximum
        assert 0 <= add_probability <= 1
        self.add_probability = add_probability
        return self

    def add_high_error_nodes(self, nodes: list, distribution, param: dict, minimum: float, maximum: float, add_probability: float):
        self.high_error_nodes = nodes
        self.he_distribution = distribution
        self.he_param = param
        self.he_minimum = minimum
        self.he_max = maximum
        self.he_add_prob = add_probability
        return self

    def add_zero_probability(self, zero_probability: float):
        self.zero_probability = zero_probability
        return self

    def error_prone_sampling(self, node_u: int, node_v: int, value: float) -> float:
        """
        Calculates the annotator-error based on the given parameters.
        """
        # Zero annotation
        if np.random.choice([True, False], p=[self.zero_probability, 1 - self.zero_probability]):
            return 0

        # nodes with high error
        if node_u in self.high_error_nodes or node_v in self.high_error_nodes:
            assert callable(self.he_distribution)

            error_sample = self.he_distribution(**self.he_param)
            coefficient = np.random.choice([1, -1], p=[self.he_add_prob, 1 - self.he_add_prob])

            return np.min([np.max([value + coefficient * error_sample, self.he_minimum]), self.he_max])

        assert callable(self.distribution)
        error_sample = self.distribution(**self.param)
        coefficient = np.random.choice([1, -1], p=[self.add_probability, 1 - self.add_probability])

        return np.min([np.max([value + coefficient * error_sample, self.minimum]), self.maximum])
