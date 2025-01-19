import train 
import test
import draw as d

"""
main file to train, test, and use models

    Train:
Train a model on the MNIST dataset by calling the train function
Saves the model and training curve plots
    train.train_mnist(hidden_size1, hidden_size2, learn_rate, epochs, model_name)
eg. train.train_mnist(128, 64, .01, 3, "new_model")

    Test:
Test a model on the MNIST dataset by calling the test function
Prints accuracy of the model on testing set
    test.test_mnist(model_name)
eg. test.test_mnist("new_model")

    Predict:
Use a trained model to predict what number you've drawn with the predict function
    d.predict(model_name)
eg. d.predict("new_model")
"""

def main():
    # train.train_mnist(512, 256, .01, 25, "High Accuracy Model")
    #test.test_mnist("High Accuracy Model")
    d.predict("High Accuracy Model")


if __name__ == "__main__":
    main()