import torch

from Network_model_lstm_rnn import network_model_lstm_rnn
from Create_sample_target import create_sample_prediction

def predict():
    # Load trained model
    model = network_model_lstm_rnn()
    model.load_state_dict(torch.load("Trained_models/trained_model_lstm_rnn.pt"))

    # Load data
    current_series, sample = create_sample_prediction("data/Train/Daily-train.csv")

    # Create tensors from data arrays
    tensor_sample = torch.from_numpy(sample).float()

    print("Tensor shape:")
    print("Sample: " + str(sample.shape))

    # Predict based on input and send to numpy
    prediction = model(tensor_sample).cpu()
    prediction = prediction.detach().numpy()

    return prediction