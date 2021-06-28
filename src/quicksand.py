import numpy as np
from collections import Counter

from scipy.stats import entropy

# sample = np.round(np.random.rand(100) * 3 + 1)
sample = [1] * 100 + [0]
print(Counter(sample))
print([v for k, v in Counter(sample).items()])
print(sum([v for k, v in Counter(sample).items()]))
print(entropy([v for k, v in Counter(sample).items()], base=2) / np.log2(len([v for k, v in Counter(sample).items()])))

print(entropy([25, 25, 25, 25], base=2)/ np.log2(4))
print(entropy([25, 25, 25, 25], base=2)/ np.log2(4))