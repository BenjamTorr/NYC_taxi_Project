import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scripts.model_utils import *
from scripts.utils import *
import random
import argparse
import numpy as np

def main(learning_rate, num_epochs, batch_size):
    """
    Simple training loop:
    This code trains a neural network to predict the fare amount of taxi drives using the drop off and pick up locations.
    The code saves the model in the folder models.

    Args:
        learning_rate: learning rate used in the optimizer of the neural networks
        num_epochs: Number of epochs for the training
        batch_size: Number of samples at a time to optimize for in each step.

    """
    # Set fixed random number seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


    # load data
    raw_df = raw_taxi_df(filename="data\\yellow_tripdata_2024-01.parquet")
    # clean data
    clean_df = clean_taxi_df(raw_df=raw_df)
    # origin location
    location_ids = ['PULocationID', 'DOLocationID']
    # Split train and test for model fitting
    X_train, X_test, y_train, y_test = split_taxi_data(clean_df=clean_df, 
                                                   x_column=location_ids, 
                                                   y_column="fare_amount", 
                                                   train_size=500000)

    # Pytorch: Load the dataset for trainging
    dataset = NYCTaxiExampleDataset(X_train=X_train, y_train=y_train)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
  
    # Initialize the MLP
    mlp = MLP(encoded_shape=dataset.X_enc_shape)
  
    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
  
    # Run the training loop
    for epoch in range(0, num_epochs): # 5 epochs at maximum
        print(f'-------------------Starting epoch {epoch+1}----------------------')
        current_loss = 0.0
    
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500) + ' in epoch ' + str(epoch + 1))
            current_loss = 0.0
   
    # Process is complete.
    print('Training process has finished.')
    # Save model
    torch.save(mlp.state_dict(), f'models/trained_model_lr_{learning_rate}_epochs_{num_epochs}.pth')
    print('Model is saved in models folder.')
    
    return X_train, X_test, y_train, y_test, data, mlp

if __name__ == "__main__":
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Train a model with specified hyperparameters")

    # Add arguments for hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of samples for the optimization')

    # Parse the command-line arguments
    
    args = parser.parse_args()
    main(args.learning_rate, args.num_epochs, args.batch_size)