import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

def log_dist_iter(coms, nodes, std_dev, threshold):
        r_nodes = nodes
        community_split_nodes = 0
        community_split_probability = lognorm.pdf(np.linspace(1, coms, coms), std_dev)

        while r_nodes > threshold:
                community_split_nodes += community_split_probability * r_nodes
                r_nodes = nodes - sum(community_split_nodes)

        community_split_nodes = np.array([int(x) if int(x) > 0 else 1 for x in community_split_nodes])
        community_split_nodes[0] += nodes - sum(community_split_nodes)
        return community_split_nodes
        

# for i in np.arange(0.1, 1.1, 0.1):
community_split = lognorm.pdf(np.linspace(1, 4, 4), 0.1) * 500
community_split[0] += 500 - sum(community_split)
print(round(0.1, 1), community_split, 500 - sum(community_split))
plt.plot(np.linspace(1, 4, 4), community_split/sum(community_split), label='0.1, old')

community_split = lognorm.pdf(np.linspace(1, 4, 4), 0.9) * 500
community_split[0] += 500 - sum(community_split)
print(round(0.9, 1), community_split, 500 - sum(community_split))
plt.plot(np.linspace(1, 4, 4), community_split/sum(community_split), label='0.9, old')

community_split = log_dist_iter(4, 500, 0.1, 5)
print(round(0.1, 1), community_split, 500 - sum(community_split))
plt.plot(np.linspace(1, 4, 4), community_split/sum(community_split), label='0.1, new')

community_split = log_dist_iter(4, 500, 0.9, 5)
print(round(0.9, 1), community_split, 500 - sum(community_split))
plt.plot(np.linspace(1, 4, 4), community_split/sum(community_split), label='0.9, new')




plt.legend(title='Legend')
plt.show()