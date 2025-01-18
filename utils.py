import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn

"""
utility functions used to save, load, and visualize models for training/testing purposes

"""

def save_model(model, file_name):
    np.savez(file_name,
            h1_size=model.h1_size,
            h2_size=model.h2_size,
            w_in_h1=model.w_in_h1,
            w_h1_h2=model.w_h1_h2,
            w_h2_out=model.w_h2_out,
            b_in_h1=model.b_in_h1,
            b_h1_h2=model.b_h1_h2,
            b_h2_out=model.b_h2_out)
    print(f"Model saved to {file_name}.npz")


def load_model(file_name):
    try:
        data = np.load(f"{file_name}.npz")
    except FileNotFoundError:
        print(f"Error: File {file_name}.npz not found.")
        return None
    
    h1_size = data['h1_size']
    h2_size = data['h2_size']
    model = nn.neural(h1_size, h2_size)
    
    model.w_in_h1 = data['w_in_h1']
    model.w_h1_h2 = data['w_h1_h2']
    model.w_h2_out = data['w_h2_out']
    model.b_in_h1 = data['b_in_h1']
    model.b_h1_h2 = data['b_h1_h2']
    model.b_h2_out = data['b_h2_out']
    
    print(f"Model loaded from {file_name}.npz")
    return model

def plot_training_curves(train_loss, train_accuracy, filename="training_curves.png"):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 5))

    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()

    # save figures in file
    plt.savefig(filename)
    print(f"Training curves saved as {filename}")

