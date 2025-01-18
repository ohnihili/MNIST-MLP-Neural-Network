import neural_network as nn
import load_data as ld
import functions as fn
import utils as ut
import numpy as np

"""
defines the train method that actualy trains a model
when called: hidden layer sizes, learn rate, and number of epochs 
can all be altered to test different models/training methods

"""

def train(hidden_size1, hidden_size2, learn_rate, epochs, model_name):

    model = nn.neural(hidden_size1, hidden_size2)
    images, labels = ld.load_mnist_train()
    
    epoch_count = 1
    correct_pred = 0

    for epoch in range(epochs):
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

            #              check accuracy of predictions
            correct_pred += int(np.argmax(output) == np.argmax(l))
            
            #                     backward propagation
            # output -> h2 
            delta_out = fn.cel_der(output,l)
            model.w_h2_out += -learn_rate * delta_out @ np.transpose(h2_act)
            model.b_h2_out += -learn_rate * delta_out

            # h2 -> hidden1
            delta_h2 = np.transpose(model.w_h2_out) @ delta_out * (h2_act * (1 - h2_act))
            model.w_h1_h2 += -learn_rate * delta_h2 @ np.transpose(h1_act)
            model.b_h1_h2 += -learn_rate * delta_h2
            
            # h1 -> input 
            delta_h1 = np.transpose(model.w_h1_h2) @ delta_h2 * (h1_act * (1 - h1_act))
            model.w_in_h1 += -learn_rate * delta_h1 @ np.transpose(img)
            model.b_in_h1 += -learn_rate * delta_h1

        print("Epoch",epoch_count, f"accuracy: {round((correct_pred / images.shape[0]) * 100, 2)}%")
        correct_pred = 0
        epoch_count += 1


    # save model
    ut.save_model(model, model_name)


