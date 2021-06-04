import pickle
import numpy as np
# from analysis.wug_metric_result import WUGMetricResult as Results

# judgements = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]

# test = [1, 2, 3, 4, 5, 6]

# results = Results()

# results.add_metric('randInd', (2, 4, len(test)))
# results.update_value('randInd', (1, 3), test)

print('ppf')

start = 0
perc = [0.1, 0.2, 0.3, 0.4]
for i, pa in enumerate(perc):
    end = start + round(pa * 111) if i < len(perc) - 1 else 111
    print(i < len(perc) - 1)
    print(start, end - 1)
    start = end