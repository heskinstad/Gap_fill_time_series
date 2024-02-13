import torch

from Network_model_lstm_rnn import network_model_lstm_rnn
from Create_sample_target import create_sample_target

def predict():
    # Load trained model
    model = network_model_lstm_rnn()
    model.load_state_dict(torch.load("Trained_models/trained_model_lstm_rnn.pt"))

    # Load data
    samples, targets = create_sample_target("data/Train/Daily-train.csv", 60, 75)

    # Create tensors from data arrays
    tensor_samples = torch.from_numpy(samples).float()
    tensor_targets = torch.from_numpy(targets).float()

    # Predict based on input and send to numpy
    prediction = model(tensor_samples).cpu()
    prediction = prediction.detach().numpy()

    print(prediction)