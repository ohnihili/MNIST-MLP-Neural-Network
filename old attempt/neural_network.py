import load_data as ld
import numpy as np


class neural:
        # constructor
    def __init__ (self, hidden_size1, hidden_size2, learning_rate):
        # set learning rate 
        self.learn = learning_rate

        # load the training data
        self.input_data, self.labels = ld.load_train()

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

    # defines forward pass/propogation
    def forward(self):
        # input to h1
            # finds the pre-activation values of each h1 node
        self.h1_pre = np.dot(self.w_i_h1,self.input_data.T) + self.b_i_h1
            # uses act-function to properly set each h1 node
        self.h1_a = self.sigmoid(self.h1_pre)

        # h1 to h2
        self.h2_pre = np.dot(self.w_h1_h2,self.h1_a) + self.b_h1_h2
        self.h2_a = self.sigmoid(self.h2_pre)

        # h2 to output
        self.out_pre = np.dot(self.w_h2_o, self.h2_a) + self.b_h2_o
        self.predictions = self.softmax(self.out_pre)
        
        # return the normalized predictions from the output layer
        return self.predictions

    # defines backpropogation
    def back(self):
        # calculate error then backprop from out to h2 | d_out_pre == deriv of loss with respect to out_pre
        error = self.cel_der(self.predictions, self.labels.T)
        d_out_pre = error * self.sigmoid_der(self.out_pre)

        # calculate gradients for w and b then backprop from h2 to h1
        grad_w_h2_o = np.dot(d_out_pre,self.h2_a.T)
        grad_b_h2_o = np.sum(d_out_pre, axis=1, keepdims=True)

        d_h2_a = np.dot(self.w_h2_o.T, d_out_pre)
        d_h2_pre = d_h2_a * self.sigmoid_der(self.h2_pre)

        # calculate gradients for w and b then backprop from h1 to inp
        grad_w_h1_h2 = np.dot(d_h2_pre, self.h1_a.T)
        grad_b_h1_h2 = np.sum(d_h2_pre, axis=1, keepdims=True)
        
        d_h1_a = np.dot(self.w_h1_h2.T, d_h2_pre)
        d_h1_pre = d_h1_a * self.sigmoid_der(self.h1_pre)

        # calculate gradients for w and b
        grad_w_i_h1 = np.dot(d_h1_pre, self.input_data)
        grad_b_i_h1 = np.sum(d_h1_pre, axis=1, keepdims=True)

        # update all w and b (gradient descent)
        self.w_i_h1 -= self.learn * grad_w_i_h1
        self.b_i_h1 -= self.learn * grad_b_i_h1
        
        self.w_h1_h2 -= self.learn * grad_w_h1_h2
        self.b_h1_h2 -= self.learn * grad_b_h1_h2
        
        self.w_h2_o -= self.learn * grad_w_h2_o
        self.b_h2_o -= self.learn * grad_b_h2_o

    # sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid
    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # cross entropy loss cost function
    def cel(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

    # derivative of cel
    def cel_der(self, y_pred, y_true):
        return y_pred - y_true

    # softmax function for normalizing output predictions
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)