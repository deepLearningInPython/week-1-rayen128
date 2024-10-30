import numpy as np
from tasks import *

inputs = np.array([[1, 2], [3, 4]])
weights = np.array([1, -1])

matrix = (np.matmul(inputs, weights))

print(matrix)
print(np.apply_along_axis(ReLu, axis=0, arr=matrix))
