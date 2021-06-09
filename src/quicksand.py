# just a script to play around, test some functionalities, find bugs
import numpy as np

a = {'a':np.array(np.random.randn(14)), 'b':np.array(np.random.randn(14)), 
        'c':np.array(np.random.randn(14)), 'd':np.array(np.random.randn(14))}

e = np.stack([v for k, v in a.items()])
print(e)
print(e.T)