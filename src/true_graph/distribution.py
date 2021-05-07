import numpy as np

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
