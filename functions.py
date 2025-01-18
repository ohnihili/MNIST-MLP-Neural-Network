import numpy as np

"""
functions used during forward and backward propagation
no need for actual cross entropy loss cost function, only derivative is required
"""

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# derivative of cross entropy loss cost function
def cel_der(output, label):
    return output - label
