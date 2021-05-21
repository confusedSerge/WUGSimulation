from visualization.metric_vis import threed_line_ploter
# # from graphs.wu_simulation_graph import WUSimulationGraph

# # # True, 'data/figs/test.png',
# bar_metric(['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'], 'Test', 'Score', 
#     metric1=[0.5, 0.7, 0.1, 0.99, 0.5, 0.7, 0.1, 0.99], 
#     metric2=[0.3, 0.3, 0.6, 0.1, 0.5, 0.7, 0.1, 0.99], 
#     metric3=[0.5, 0.9, 0.5, 0.9, 0.5, 0.7, 0.1, 0.99],
#     metric4=[0.7, 0.1, 0.3, 0.4, 0.5, 0.7, 0.1, 0.99])

# line_ploter(['0%', '20%', '40%', '60%', '80%', '100%'], 'Test', 'Percentage', 'Score', 
#     func1=[0.5, 0.7, 0.8, 0.99, 0.99, 0.99], 
#     func2=[0.3, 0.4, 0.6, 0.6, 0.7, 0.7], 
#     func3=[0.5, 0.9, 0.8, 0.9, 0.8, 0.9],
#     func4=[0.2, 0.2, 0.3, 0.4, 0.3, 0.6])

# import numpy as np
# from collections import Counter

# s = np.random.poisson(0.2, 10000)
# print(Counter(s))

# import matplotlib.pyplot as plt

# count, bins, ignored = plt.hist(s, 4, density=True)

# plt.show()

# print(np.max([1, 2]))

# threed_line_ploter(['20%', '40%', '60%', '80%'], 'n', 
#     [1, 2, 3], 'k', 'Performance', 'Wow...', 
#     metric1={1: [0.5, 0.7, 0.8, 0.99], 2: [0.1, 0.2, 0.3, 0.1], 3: [0.8, 0.7, 0.6, 0.15]},
#     metric2={1: [0.1, 0.2, 0.3, 0.4], 2: [0.5, 0.99, 0.8, 0.99], 3: [0.16, 0.3, 0.5, 0.7]},
#     metric3={1: [0.3, 0.4, 0.8, 0.99], 2: [0.1, 0.46, 0.546, 0.4], 3: [0.2, 0.7, 0.5, 0.3]},
#     metric4={1: [0.2, 0.7, 0.8, 0.99], 2: [0.1, 0.2, 0.3, 0.1], 3: [0.8, 0.7, 0.6, 0.15]}
#     )
