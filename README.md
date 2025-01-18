# MNIST-MLP-Neural-Network

## Project Overview:

Implementation of a Multilayer Perceptron Neural Network from scratch (only using numpy), trained on the MNIST DataSet, to classify hand drawn digits.

User can choose between three main methods within main.py:

1. Train: Trains a new model on the 60,000 training data points from MNIST
2. Test: Tests a trained a model on the 10,000 testing data points from MNIST
3. Predict: Opens a drawing window where the user can draw numbers for a chosen model to predict.

A ~98% accuracy model is included to use with predict immediately

---

## Instructions to Run the Code:

### 1. Requirements:
   - Check requirements.txt

### 2. Running each method:

Train:
Saves both a trained model in a _.npz file as well as a plot of the accuracy and loss during training in "training_curves.png"

    train.train_mnist(hidden_size1, hidden_size2, learn_rate, epochs, model_name)
eg. 
    
    train.train_mnist(128, 64, .01, 3, "new_model")

Test:
Prints accuracy of the model on testing set

    test.test_mnist(model_name)
eg. 

    test.test_mnist("new_model")

Predict:

    d.predict(model_name)
eg. 

    d.predict("new_model")

---

## Known Issues:

- Do not run both 'test.test_mnist' and 'd.predict' in the same main call as the script will not end unless forcefully closed

---
