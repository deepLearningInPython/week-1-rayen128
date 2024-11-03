import numpy

# Task 1:
# Instructions:
# Write a function that takes one numeric argument as input.
# If the number is larger than zero, the function should return 1, otherwise is should return -1.
# The name of the function should be step


def step(value: int) -> int:
    if value > 0:
        return 1
    else:
        return -1


# Task 2:
# Instructions:
# Write a function that takes in two arguments: a numpy array, and an integer (call argument "cutoff" and set default to 0).
# The function should return a numpy array of the same length, with all elements smaller than the cutoff being set to cutoff).
# The name of the function should be ReLu

def ReLu(array, cutoff: int = 0):

    copy = numpy.copy(array)

    copy[copy < cutoff] = cutoff
    return copy


# Task 3:
# Instruction:
# Write a function that takes in a two-dimensional numpy array of size (n, p) and a one-dimensional numpy array of size p.
# The function should start by multiplying the two numpy arrays (matrix multiplication).
# Next, apply the ReLu function from above to the resulting matrix and return the result.
# Name the function neural_net_layer

# Your code here:

def neural_net_layer(input, weights):
    matrix = numpy.matmul(input, weights)
    matrix = ReLu(matrix)
    return matrix
