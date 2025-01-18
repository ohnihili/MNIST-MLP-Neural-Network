import numpy as np
import neural_network as nn
import load_data as ld
import utils as ut

# train model
def train_model(model, epochs):
    # Iterate over the epochs
    for epoch in range(epochs):
        # Track the loss for the current epoch
        epoch_loss = 0

        # Iterate over the training data one sample at a time
        for i in range(len(model.input_data)):  # Access data directly from the model
            # Get the current input and label
            x_sample = model.input_data[i:i+1]  # Make it a 2D array (1, 784)
            y_sample = model.labels[i:i+1]  # Same for label

            # Update model's input_data with the current sample
            model.input_data = x_sample

            # Perform forward pass and get predictions
            predictions = model.forward()

            # Compute loss for the sample
            loss = model.cel(predictions, y_sample.T)
            epoch_loss += loss

            # Perform backward pass and update the weights
            model.back()

        # Print loss for the current epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(model.input_data)}")

        # After each epoch, save the model
        ut.save_model(model, f"model_epoch_{epoch+1}.npz")

# save model 
# ut.save_model(model, "trained_model.npz")

# # To load the model later
# model = ut.load_model("trained_model.npz", model)