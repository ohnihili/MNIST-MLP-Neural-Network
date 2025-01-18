import train 
import test

"""
main file to train and test models

"""

def main():
    train.train(20,15,0.01,2,"Trained Model")
    test.test("Trained Model")

if __name__ == "__main__":
    main()