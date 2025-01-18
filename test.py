import load_data as ld
import functions as fn
import utils as ut
import numpy as np

"""
defines the test method that tests a trained model
"""

def test(model_name):
    model = ut.load_model(model_name)

    images, labels = ld.load_mnist_test()

    correct_pred = 0

    for i in range(1):
        for img,l in zip(images, labels):
            img.shape += (1,)
            l.shape += (1,)
            
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

            #             check accuracy of predictions
            correct_pred += int(np.argmax(output) == np.argmax(l))
            
        print(f"Testing accuracy: {round((correct_pred / images.shape[0]) * 100, 2)}%")

