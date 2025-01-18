import train
import test
import load_data as ld
import utils
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train or Test the neural network.")
    utils.save_img(10,2,5)

    
    images, labels = ld.load_train()
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()