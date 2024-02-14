import numpy as np
import torch

import Parameters
from Network_model_lstm_rnn import network_model_lstm_rnn
from Create_sample_target import create_sample_prediction

def predict(sample=np.ones(1)):
    # Load trained model
    model = network_model_lstm_rnn()
    model.load_state_dict(torch.load(Parameters.path_trained_model))

    # Load data
    if sample.all() == 1:
        _, sample = create_sample_prediction(Parameters.path_test_data)

    # Create tensors from data arrays
    tensor_sample = torch.from_numpy(sample).float()

    print("Tensor shape:")
    print("Sample: " + str(sample.shape))

    # Predict based on input and send to numpy
    prediction = model(tensor_sample).cpu()
    prediction = prediction.detach().numpy()

    return prediction


def predict_multiple():
    current_series, _ = create_sample_prediction(Parameters.path_test_data)
    predicted_series = current_series.copy()

    for i in range(Parameters.length_of_prediction):
        predicted_series[Parameters.series_prediction_start+i] = predict(
            predicted_series[Parameters.series_prediction_start+i-Parameters.lookback:
                           Parameters.series_prediction_start+i])

    for i in range(Parameters.series_prediction_start):
        predicted_series[i] = np.nan
    for i in range(Parameters.series_prediction_start+Parameters.length_of_prediction, len(predicted_series)):
        predicted_series[i] = np.nan

    return current_series, predicted_series

