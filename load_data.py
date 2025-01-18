import numpy as np
import pathlib

"""
loads and processes data for training/testing
returns:
    images (784,1): normalized images, flattened as 1 pixel for each input node
    labels (10,1): actual img labels as ints

"""

# loads some data for meta testing purposes
def load_for_test():
     with np.load(f"mnist.npz") as f:
        images, labels = f['x_test'], f['y_test']
        return images, labels
     

def one_hot_encode(labels):
    # initialize an array of zeros with shape (num_samples, 10)
    one_hot_labels = np.zeros((labels.size, 10))
    
    # set the corresponding index to 1 for each label
    one_hot_labels[np.arange(labels.size), labels] = 1
    
    return one_hot_labels


# loads the training data
def load_train():
    with np.load(f"mnist.npz") as f:
        images, labels = f['x_train'], f['y_train']

        # normalize pixel values between [0, 1]
        images = images.astype(np.float32) / 255.0

        # flatten the images from 28x28 to 784,1
        images = images.reshape(images.shape[0], -1)

        labels = one_hot_encode(labels)

        return images, labels

# loads the testing data
def load_test():
    with np.load(f"mnist.npz") as f:
        images, labels = f['x_test'], f['y_test']
        
        # normalize pixel values between [0, 1]
        images = images.astype(np.float32) / 255.0

        # flatten the images from 28x28 to 784,1
        images = images.reshape(images.shape[0], -1)
        
        return images, labels


(x_train, y_train) = load_train()