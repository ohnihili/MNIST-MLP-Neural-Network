import numpy as np
import pathlib

def load_train():
    with np.load(f"mnist.npz") as f:
        images, labels = f['x_train'], f['y_train']
        # normalize the images

        return images, labels

def load_test():
    with np.load(f"mnist.npz") as f:
        images, labels = f['x_test'], f['y_test']
        # normalize the images
        
        return images, labels


(x_train, y_train) = load_train()