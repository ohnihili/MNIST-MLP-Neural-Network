import neural_network as nn
import load_data as ld
import functions as fn
import utils as ut
import numpy as np

"""


"""

def main():

    model = nn.neural(hidden_size1=128, hidden_size2=64)
    images, labels = ld.load_mnist()

    learn_rate = 0.01
    epochs = 1
    nr_correct = 0

    for epoch in range(epochs):
        for img,l in zip(images, labels):
            img.shape += (1,)
            l.shape += (1,)
            
            #             call forward pass
            # input to h1
            h1_pre = model.b_in_h1 + model.w_in_h1 @ img
            h1_act = fn.sigmoid(h1_pre)

            # h1 to h2
            h2_pre = model.b_h1_h2 + model.w_h1_h2 @ h1_act
            h2_act = fn.sigmoid(h2_pre)

            # h2 to output
            out_pre = model.b_h2_out + model.w_h2_out @ h2_act
            output = fn.sigmoid(out_pre)

            #              call error functions derivative
            error = fn.cel(output,l)
            nr_correct += int(np.argmax(output) == np.argmax(l))
            
            #                  call backpropagation
            # Backpropagation output -> hidden2 (cost function derivative)
            delta_out = fn.cel_der(output,l)
            model.w_h2_out += -learn_rate * delta_out @ np.transpose(h2_act)
            model.b_h2_out += -learn_rate * delta_out

            # Backpropagation hidden2 -> hidden1 (activation function derivative)
            delta_h2 = np.transpose(model.w_h2_out) @ delta_out * (h2_act * (1 - h2_act))
            model.w_h1_h2 += -learn_rate * delta_h2 @ np.transpose(h1_act)
            model.b_h1_h2 += -learn_rate * delta_h2
            
            # Backpropagation hidden1 -> input (activation function derivative)
            delta_h1 = np.transpose(model.w_h1_h2) @ delta_h2 * (h1_act * (1 - h1_act))
            model.w_in_h1 += -learn_rate * delta_h1 @ np.transpose(img)
            model.b_in_h1 += -learn_rate * delta_h1

        print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
        nr_correct = 0


    # save model
    # ut.save_model(model)
    return


if __name__ == "__main__":
    main()