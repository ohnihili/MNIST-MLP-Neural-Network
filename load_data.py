import numpy as np

"""
loads and processes data for training/testing
returns:
   images (784,1): normalized images, flattened as 1 pixel for each input node
   labels (10,1): actual img labels as ints
"""

# loads mnist training data 
def load_mnist_train():
    with np.load(f"mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]

    # normalize images
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

    #one hot encode the labels
    labels = np.eye(10)[labels]

    return images, labels

# loads mnist testing data 
def load_mnist_test():
    with np.load(f"mnist.npz") as f:
        images, labels = f["x_test"], f["y_test"]

    # normalize images
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

    #one hot encode the labels
    labels = np.eye(10)[labels]

    return images, labels