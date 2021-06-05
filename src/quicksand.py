# just a script to play around, test some functionalities, find bugs
import numpy as np

a = np.zeros((3, 3))
print(a)

a = np.expand_dims(a, axis=0)
print('Extended \n', a)
print(a.shape)
a = np.append(a, [[[1, 2, 3,], [4, 5, 6,], [7, 8, 9]]], axis=1)
print('Appended \n', a)