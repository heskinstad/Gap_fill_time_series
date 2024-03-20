import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class network_model_lstm_rnn(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=300, output_size=50, num_layers=1):
        super(network_model_lstm_rnn, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, padded_samples, samples_lengths, padded_targets=None, targets_lengths=None):        # Pack the input sequence
        #packed_input = pack_padded_sequence(input_seq, seq_lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, padded_samples.size(0), self.hidden_layer_size).to(padded_samples.device)
        c0 = torch.zeros(self.num_layers, padded_samples.size(0), self.hidden_layer_size).to(padded_samples.device)

        # Forward pass through LSTM layer
        packed_output, (hn, cn) = self.lstm(padded_samples, (h0, c0))

        # If you want to work with the output as a padded sequence
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # Assuming you're only interested in the final output for each sequence for simplicity
        # This might be different based on your specific use case
        # For instance, you may want to run the output through another layer that supports variable lengths
        # or handle variable-length predictions differently
        final_outputs = self.linear(output)

        return final_outputs
