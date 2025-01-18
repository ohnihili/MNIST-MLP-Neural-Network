import numpy as np
import functions as fn

"""


"""

def forward(model, img):
    # input to h1
    h1_pre = model.b_in_h1 + model.w_in_h1 @ img
    h1_a = fn.sigmoid(h1_pre)

    # h1 to h2
    h2_pre = model.b_h1_h2 + model.w_h1_h2 @ h1_a
    h2_a = fn.sigmoid(h2_pre)

    # h2 to output
    out_pre = model.b_h2_out + model.w_h2_out @ h2_a
    output = fn.sigmoid(out_pre)
    
    # return the normalized predictions from the output layer
    return output