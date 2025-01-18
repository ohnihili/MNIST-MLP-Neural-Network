import new_load_data as ld
import numpy as np


class neural:
    # constructor
    def __init__ (self, hidden_size1, hidden_size2):
        # set the number of nodes in each layer
        self.input_size= 784
        self.hidden1 = hidden_size1
        self.hidden2 = hidden_size2
        self.output_size = 10

        # initialize the weights between layers | w_h_i -> Weights from Input layer to Hidden layer 1
        self.w_i_h1 = np.random.uniform(-0.5,.05, (self.hidden1,self.input_size))
        self.w_h1_h2 = np.random.uniform(-0.5,.05, (self.hidden2,self.hidden1))
        self.w_h2_o  = np.random.uniform(-0.5,.05, (self.output_size,self.hidden2))

        # initialize biases
        self.b_i_h1 = np.zeros((self.hidden1,1))
        self.b_h1_h2 = np.zeros((self.hidden2,1))
        self.b_h2_o = np.zeros((self.output_size,1))
