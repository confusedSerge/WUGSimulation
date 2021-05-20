

class Annotator():
    
    def __init__(self, distribution, *args) -> None:
        """
        Initializes an Annotator with a certaint distribution,
            based on which an error is calculated

        We expect sth like:
            Annotator(np.random.poisson, 0.2)
        """
        self.distribution = distribution
        self.args = args

    def sample_error(self) -> int:
        """
        Calculates the annotator-error based on the given distribution
        """
        return self.distribution(*args)