import numpy as np

"""


"""

# defines backpropogation
def back(error_der, model, img, learn_rate):
    # calculate error then backprop from out to h2 | d_out_pre == deriv of loss with respect to out_pre
    model.w_h2_o += -learn_rate * delta_o @ np.transpose(h)
    model.b_h2_o += -learn_rate * delta_o
    d_out_pre = error * sigmoid_der(out_pre)

    # calculate gradients for w and b then backprop from h2 to h1
    grad_w_h2_o = np.dot(d_out_pre,h2_a.T)
    grad_b_h2_o = np.sum(d_out_pre, axis=1, keepdims=True)

    d_h2_a = np.dot(w_h2_o.T, d_out_pre)
    d_h2_pre = d_h2_a * sigmoid_der(h2_pre)

    # calculate gradients for w and b then backprop from h1 to inp
    grad_w_h1_h2 = np.dot(d_h2_pre, h1_a.T)
    grad_b_h1_h2 = np.sum(d_h2_pre, axis=1, keepdims=True)
    
    d_h1_a = np.dot(w_h1_h2.T, d_h2_pre)
    d_h1_pre = d_h1_a * sigmoid_der(h1_pre)

    # calculate gradients for w and b
    grad_w_i_h1 = np.dot(d_h1_pre, input_data)
    grad_b_i_h1 = np.sum(d_h1_pre, axis=1, keepdims=True)

    # update all w and b (gradient descent)
    w_i_h1 -= learn * grad_w_i_h1
    b_i_h1 -= learn * grad_b_i_h1
    
    w_h1_h2 -= learn * grad_w_h1_h2
    b_h1_h2 -= learn * grad_b_h1_h2
    
    w_h2_o -= learn * grad_w_h2_o
    b_h2_o -= learn * grad_b_h2_o
