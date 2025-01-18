import matplotlib.pyplot as plt
import load_data as ld
import numpy as np

def save_img(n, rows, cols):
    images, labels = ld.load_for_test()
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("mnist_images.png")



def save_model(model, file_name):
    np.savez(file_name,
            w_i_h1=model.w_i_h1,
            w_h1_h2=model.w_h1_h2,
            w_h2_o=model.w_h2_o,
            b_i_h1=model.b_i_h1,
            b_h1_h2=model.b_h1_h2,
            b_h2_o=model.b_h2_o)
    print(f"Model saved to {file_name}")


def load_model(file_name, model):
    data = np.load(file_name)
    
    model.w_i_h1 = data['w_i_h1']
    model.w_h1_h2 = data['w_h1_h2']
    model.w_h2_o = data['w_h2_o']
    model.b_i_h1 = data['b_i_h1']
    model.b_h1_h2 = data['b_h1_h2']
    model.b_h2_o = data['b_h2_o']
    
    print(f"Model loaded from {file_name}")
    
    return model

