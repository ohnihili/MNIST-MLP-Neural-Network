import matplotlib.pyplot as plt
import load_data as ld

def save_img(n, rows, cols):
    images, labels = ld.load_train()
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("mnist_images.png")
    return


def save_model():
    return

def load_model():
    return

