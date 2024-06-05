import torch
from torch import nn

import Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class network_model_lstm_rnn(nn.Module):
    def __init__(self, input_size=Parameters.input_size, hidden_layer_size=Parameters.hidden_layer_size, output_size=Parameters.network_output_size, num_layers=Parameters.num_layers):
        super(network_model_lstm_rnn, self).__init__()
        self.hidden_layer_size = hidden_layer_size  # Larger hidden layer makes it slower but leads to better predictions (but also easier overfitting - be careful!)

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        # Input is of type (batch_size, sequence_length, features) which means we need to use batch_first=True

        # Fully connected layer which reduces the output size to the size of the gap (otherwise the dims of our hidden layer would be our number of outputs)
        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, input_seq):
        # Initialize hidden state
        h0 = torch.zeros(Parameters.num_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_().to(device)
        c0 = torch.zeros(Parameters.num_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_().to(device)

        # Forward pass
        lstm_out, (hn, cn) = self.lstm(input_seq, (h0.detach(), c0.detach()))

        lstm_out = lstm_out[:, -1, :]

        predictions = self.linear(lstm_out)

        # Unsqueeze to reshape correctly
        predictions = predictions.unsqueeze(-1)

        return predictions