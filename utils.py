import matplotlib.pyplot as plt
import numpy as np

"""


"""

def save_model(model, file_name):
    np.savez(file_name,
            w_i_h1=model.w_i_h1,
            w_h1_h2=model.w_h1_h2,
            w_h2_o=model.w_h2_o,
            b_i_h1=model.b_i_h1,
            b_h1_h2=model.b_h1_h2,
            b_h2_o=model.b_h2_o)
    print(f"Model saved to {file_name}")


def load_model(file_name, model):
    data = np.load(file_name)
    
    model.w_i_h1 = data['w_i_h1']
    model.w_h1_h2 = data['w_h1_h2']
    model.w_h2_o = data['w_h2_o']
    model.b_i_h1 = data['b_i_h1']
    model.b_h1_h2 = data['b_h1_h2']
    model.b_h2_o = data['b_h2_o']
    
    print(f"Model loaded from {file_name}")
    
    return model

