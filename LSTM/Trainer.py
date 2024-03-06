import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from LSTM import Parameters


class Trainer:
    def __init__(self, model, dataset, optimizer, loss_fn):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def train(self, epochs):
        learning_rate = self.optimizer.param_groups[0]['lr']

        loss_values = np.empty(epochs, dtype=float)

        # Define the weight factor for the edges
        edge_weight = 10  # for example, the edges are 10 times more important

        # Adjust these indices based on where the gap starts and ends in your input sequences
        gap_start = 50
        gap_end = 99

        def create_gap_weights(batch_size, sequence_length, gap_start, gap_end, edge_weight):
            # Create an array of ones
            weights = np.ones((batch_size, sequence_length, 1))

            # Assign higher weights to the edges of the gap
            weights[:, gap_start:gap_start + edge_weight, :] = edge_weight
            weights[:, gap_end - edge_weight:gap_end, :] = edge_weight

            return torch.tensor(weights, dtype=torch.float32)

        for epoch in range(epochs):
            # Train on the training data
            for batch_idx, (samples, targets) in enumerate(self.dataset):
                # Forward pass
                predictions = self.model(samples.to(self.device))

                # Calculate the loss
                #loss = self.loss_fn(predictions.to(self.device), targets.to(self.device))

                # Generate weights for this batch
                batch_size = samples.size(0)  # assuming samples is a tensor of shape (batch_size, 150, 1)
                weights = create_gap_weights(batch_size, targets.size(1), gap_start, gap_end, edge_weight).to(
                    self.device)

                # Calculate the weighted loss
                loss = (weights * (predictions.to(self.device) - targets.to(self.device)) ** 2).mean()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_values[epoch] = loss

                # Learning rate that decays linearly (comment out this block if you want a constant learning rate)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate - (learning_rate / epochs) * epoch

                # Print the loss
                if batch_idx % 100 == 0:
                    print('Epoch: {} | Batch: {} | Loss: {}'.format(epoch, batch_idx, loss.item()))

        # Plot the loss
        plt.plot(loss_values)
        plt.xlabel("Loss")
        plt.savefig(os.getcwd() + r"\Trained_models\trained_model_lstm_rnn_figure.png")
        plt.show()