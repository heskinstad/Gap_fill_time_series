import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import Parameters


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

        edge_weight = 20  # Edge data points gets a weight 20 times greater than any other point

        # Create the weights of every data point in the gap. First and last value gets weight of 20
        def create_gap_weights(batch_size, sequence_length, edge_weight):
            weights = np.ones((batch_size, sequence_length, 1))

            weights[:, 0, :] = edge_weight
            weights[:, Parameters.length_of_prediction-1, :] = edge_weight

            return torch.tensor(weights, dtype=torch.float32)

        for epoch in range(epochs):
            # Train on the training data
            for batch_idx, (samples, targets) in enumerate(self.dataset):
                # Forward pass
                predictions = self.model(samples.to(self.device))

                # Calculate the loss (old loss function without weights)
                #loss = self.loss_fn(predictions.to(self.device), targets.to(self.device))

                # Generate weights for this batch
                batch_size = samples.size(0)  # assuming samples is a tensor of shape (batch_size, 150, Parameters.input_size)
                weights = create_gap_weights(batch_size, targets.size(1), edge_weight).to(self.device)

                # Calculate the weighted loss
                loss = (weights * (predictions.to(self.device) - targets.to(self.device)) ** 2).mean()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_values[epoch] = loss

                # Learning rate that decays linearly (comment out this block for a constant learning rate)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate - (learning_rate / epochs) * epoch

                # Print the loss
                if batch_idx % 100 == 0:
                    print('Epoch: {} | Batch: {} | Loss: {}'.format(epoch, batch_idx, loss.item()))

            # Save backup
            if Parameters.make_backup and epoch % 250 == 0 and epoch != 0:
                torch.save(self.model.state_dict(), os.getcwd() + r'\Model_bkps\model_bkp' + str(epoch) + '.pt')

        # Plot the loss
        plt.plot(loss_values)
        plt.xlabel("Loss")
        plt.savefig(os.getcwd() + r"\Trained_models\\" + Parameters.model_name + "_figure.png")
        plt.show()

        return loss_values[-1]