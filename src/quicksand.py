import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

lam = list(np.arange(0, 1.1, 0.1))
samples = 10**5

for _lambda_param in lam:
    _lambda_param = round(_lambda_param, 2)
    res = np.random.poisson(_lambda_param, samples)
    res = [np.min([a, 3]) for a in res]
    res = Counter(res)
    print(_lambda_param, res)

    x = []
    y = []
    for k, v in sorted(res.items(), key=lambda pair: pair[0]):
        x.append(k)
        y.append(v / samples)

    plt.plot(x, y, label=_lambda_param)

plt.legend()
plt.show()
