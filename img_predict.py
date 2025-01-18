import functions as fn
import utils as ut

"""
function for single image prediction, used in drawing prediction function
"""

def predict(img, model_name):
    model = ut.load_model(model_name)
    img.shape += (1, )
    
    #           forward propagation
    # input -> h1
    h1_pre = model.b_in_h1 + model.w_in_h1 @ img
    h1_act = fn.sigmoid(h1_pre)

    # h1 -> h2
    h2_pre = model.b_h1_h2 + model.w_h1_h2 @ h1_act
    h2_act = fn.sigmoid(h2_pre)

    # h2 -> output
    out_pre = model.b_h2_out + model.w_h2_out @ h2_act
    output = fn.sigmoid(out_pre)

    return output