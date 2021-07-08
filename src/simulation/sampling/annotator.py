import numpy as np

class Annotator():
    
    def __init__(self, distribution, param: list, minimum: float, maximum: float, add_probability: float) -> None:
        """
        Initializes an Annotator with a certain distribution,
            based on which an error is calculated

        We expect sth like:
            Annotator(np.random.poisson, 0.2, 1)
        """
        self.distribution = distribution
        self.param = param
        self.minimum = minimum
        self.maximum = maximum
        assert 0 <= add_probability <= 1
        self.add_probability = add_probability

    def error_prone_sampling(self, value) -> int:
        """
        Calculates the annotator-error based on the given function
        """
        error_sample = self.distribution(*self.param)
        coefficient = np.random.choice([1, -1], p=[self.add_probability, 1 - self.add_probability])
        
        return np.min([np.max([value + coefficient * error_sample, self.minimum]), self.maximum])
