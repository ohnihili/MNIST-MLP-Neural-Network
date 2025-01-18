import numpy as np

"""
class implementation of the network itself
w_xx_yy = weights from xx to yy
b_xx_yy = biases from xx to yy
"""
class neural:
    # constructor
    def __init__ (self, hidden_size1, hidden_size2):
        # set the number of nodes in each layer
        self.in_size = 784
        self.h1_size = hidden_size1
        self.h2_size = hidden_size2
        self.out_size = 10

        # initialize the weights between layers | w_h_i -> Weights from Input layer to Hidden layer 1
        self.w_in_h1 = np.random.uniform(-0.5, 0.5, (self.h1_size,self.in_size))
        self.w_h1_h2 = np.random.uniform(-0.5, 0.5, (self.h2_size,self.h1_size))
        self.w_h2_out  = np.random.uniform(-0.5, 0.5, (self.out_size,self.h2_size))

        # initialize biases
        self.b_in_h1 = np.zeros((self.h1_size, 1))
        self.b_h1_h2 = np.zeros((self.h2_size, 1))
        self.b_h2_out = np.zeros((self.out_size, 1))


