import numpy as np


# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# cross entropy loss cost function
def cel(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

# derivative of cel
def cel_der(y_pred, y_true):
    return y_pred - y_true

# softmax function for normalizing output predictions
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)