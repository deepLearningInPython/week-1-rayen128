import numpy as np
from tasks import *

inputs = np.array([[1, 2], [3, 4]])
weights = np.array([1, -1])

matrix = (np.matmul(inputs, weights))


print(np.apply_along_axis(ReLu, axis=0, arr=matrix))

print(ReLu(matrix))
