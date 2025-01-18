import numpy as np

"""


"""

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# cross entropy loss cost function
def cel(output, label):
    return -np.mean(label * np.log(output + 1e-8) + (1 - label) * np.log(1 - output + 1e-8))

# derivative of cel
def cel_der(output, label):
    return output - label
