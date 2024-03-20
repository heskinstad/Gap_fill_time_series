import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from LSTM import Parameters
from LSTM.Dataset_loader import custom_collate_fn


class Trainer:
    def __init__(self, model, dataset, optimizer, loss_fn):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def train(self, epochs):
        loss_values = np.empty(epochs, dtype=float)

        #data_loader = DataLoader(self.dataset, batch_size=Parameters.batch_size, shuffle=True, collate_fn=custom_collate_fn)

        for epoch in range(epochs):
            for padded_sequences, padded_targets in self.dataset:
                # Assuming your model and loss function can handle variable-sized targets directly
                predictions = self.model(padded_sequences)
                loss = self.loss_fn(predictions, padded_targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Save backup
            if Parameters.make_backup and epoch % 500 == 0 and epoch != 0:
                torch.save(self.model.state_dict(), os.getcwd() + r'\Model_bkps\model_bkp' + str(epoch) + '.pt')

        # Plot the loss
        plt.plot(loss_values)
        plt.xlabel("Loss")
        plt.savefig(os.getcwd() + r"\Trained_models\trained_model_lstm_rnn_figure.png")
        plt.show()