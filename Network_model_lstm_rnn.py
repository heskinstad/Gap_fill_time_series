import torch
from torch import nn, relu

import Parameters


'''class network_model_lstm_rnn(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=Parameters.lookback, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
'''


class network_model_lstm_rnn(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1):
        super(network_model_lstm_rnn, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # Fully connected layer to get the output size to 10 (for 10 measurements)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Initializing hidden state for first input using method defined below
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).requires_grad_()
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).requires_grad_()

        # Forward pass through LSTM layer
        # lstm_out: tensor of shape (batch_size, seq_length, hidden_layer_size)
        # hn, cn: tensors of shape (num_layers, batch_size, hidden_layer_size), containing the hidden and cell state of the last timestep
        lstm_out, (hn, cn) = self.lstm(input_seq, (h0.detach(), c0.detach()))

        # Only take the output from the final timestep
        # You might need to modify this, depending on how you want to structure your data
        lstm_out = lstm_out[:, -1, :]

        # Pass the final output of the LSTM to the linear layer
        predictions = self.linear(lstm_out)

        # Reshape predictions to match the target tensor shape ([batch_size, 1, 1])
        predictions = predictions.unsqueeze(-1)

        return predictions