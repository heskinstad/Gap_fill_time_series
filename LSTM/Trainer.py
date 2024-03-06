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

        for epoch in range(epochs):
            # Train on the training data
            for batch_idx, (samples, targets) in enumerate(self.dataset):
                # Forward pass
                predictions = self.model(samples.to(self.device))

                # Calculate the loss
                loss = self.loss_fn(predictions.to(self.device), targets.to(self.device))

                '''boundary_weight = 10.0

                # New loss calculation to improve boundary predictions
                loss = self.loss_fn(predictions.to(self.device), targets.to(self.device))  # Calculate the base loss for all points

                # Identify boundary points and apply increased weight
                # Assuming the first and last points in each sequence are boundaries, and -10 indicates a gap
                boundary_mask = torch.zeros_like(targets.to(self.device))
                boundary_mask[:, 0] = boundary_weight  # First point in each sequence
                boundary_mask[:, -1] = boundary_weight  # Last point in each sequence
                if Parameters.normalize_values:
                    boundary_mask[targets == 0.0] = boundary_weight  # Gap points
                else:
                    boundary_mask[targets == -10.0] = boundary_weight  # Gap points

                # Apply the boundary mask
                weighted_loss = loss * boundary_mask
                final_loss = weighted_loss.mean()  # Average the loss

                # Backward pass
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()'''

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