import pickle
import numpy as np
from analysis.wug_metric_result import WUGMetricResult as Results

judgements = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]

test = [1, 2, 3, 4, 5, 6]

results = Results()

results.add_metric('randInd', (2, 4, len(test)))
results.update_value('randInd', (1, 3), test)

print('ppf')