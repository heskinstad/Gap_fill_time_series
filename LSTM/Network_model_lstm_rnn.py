import torch
from torch import nn

import Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class network_model_lstm_rnn(nn.Module):
    def __init__(self, input_size=Parameters.input_size, hidden_layer_size=Parameters.hidden_layer_size, output_size=Parameters.network_output_size, num_layers=Parameters.num_layers):
        super(network_model_lstm_rnn, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # Fully connected layer to reduce the output size
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Initializing hidden state for first input using method defined below
        h0 = torch.zeros(Parameters.num_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_().to(device)
        c0 = torch.zeros(Parameters.num_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_().to(device)

        # Forward pass through LSTM layer
        lstm_out, (hn, cn) = self.lstm(input_seq, (h0.detach(), c0.detach()))

        # Reduce the output dimension size (from defined input size to defined output size)
        lstm_out = lstm_out[:, -1, :]

        # Pass the final output of the LSTM to the linear layer
        predictions = self.linear(lstm_out)

        # Reshape predictions to match the target tensor shape ([batch_size, 1, 1])
        predictions = predictions.unsqueeze(-1)

        return predictions