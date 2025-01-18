import numpy as np



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
