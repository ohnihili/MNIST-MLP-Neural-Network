import neural_network as nn
import new_load_data as ld
from forward_pass import forward
from backpropagate import back
import functions as fn
import utils as ut

def main():

    model = nn.neural(hidden_size1=128, hidden_size2=64)
    images, labels = ld.load_mnist()

    learn_rate = 0.01
    epochs = 5


    for epoch in range(epochs):
        for img,l in zip(images, labels):
            img.shape += (1,)
            l.shape += (1,)

            # call forward pass
            forward()

            # call error function
            fn.cel()

            # call backpropagation
            back()

    # save model
    ut.save_model(model)


if __name__ == "__main__":
    main()