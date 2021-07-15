import numpy as np
from graphs.base_graph import BaseGraph

a = np.array([[BaseGraph(), BaseGraph()], [BaseGraph(), BaseGraph()]])
c = a.copy()
it = np.nditer(np.zeros(c.shape), flags=['multi_index'])

print(a)
print(c)
for x in it:
    c[it.multi_index] = c[it.multi_index].get_number_nodes()

    print(c)
print(a)
print(c)
