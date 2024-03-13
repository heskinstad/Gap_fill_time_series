import torch
import torch.nn as nn

class model_encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(model_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden

class model_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(model_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, _ = self.lstm(x, (hidden, torch.zeros_like(hidden)))
        output = self.out(output)
        return output


class network_model_seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(network_model_seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, inputs, targets, target_len):
        # Encoder
        encoder_hidden = self.encoder(inputs)

        # Decoder
        # Initialize decoder input (e.g., with the value just before the gap)
        decoder_input = inputs[:, -1, :].unsqueeze(1)  # Assuming last known value before gap as initial input
        outputs = torch.zeros(inputs.size(0), target_len, targets.size(2)).to(self.device)

        for t in range(target_len):
            decoder_output = self.decoder(decoder_input, encoder_hidden)
            outputs[:, t, :] = decoder_output
            decoder_input = decoder_output  # Use own predictions as inputs for next step

        return outputs
