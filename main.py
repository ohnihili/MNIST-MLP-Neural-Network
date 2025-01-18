import neural_network as nn
import train
import test
import load_data as ld
import utils
import argparse

def main():
    # parser = argparse.ArgumentParser(description="Train or Test the neural network.")
    # utils.save_img(10,2,5)

    # images, labels = ld.load_train()
    # print(f"Images shape: {images.shape}")
    # print(f"Labels shape: {labels.shape}")

    model = nn.neural(hidden_size1=128, hidden_size2=64, learning_rate=0.01)

    train.train_model(model, 5,)


if __name__ == "__main__":
    main()