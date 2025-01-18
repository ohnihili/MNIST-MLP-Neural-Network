import train 
import test

"""
main file to train and test models

"""

def main():
    #train.train(128,64,0.01, 5,"mnist_model")
    #train.train(522,348,0.01,20,"99% Accuracy Model")
    test.test("97.88% Accuracy Model")

if __name__ == "__main__":
    main()