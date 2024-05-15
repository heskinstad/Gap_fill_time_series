import numpy as np
import torch

import Parameters
from Network_model_lstm_rnn import network_model_lstm_rnn
from Create_sample_target import create_sample_prediction, create_sample_gap_prediction
from Normalize import Denormalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(sample=None):
    # Load trained model
    model = network_model_lstm_rnn().to(device)
    model.load_state_dict(torch.load(Parameters.path_trained_model, map_location=device))

    # Load data
    if sample is None:
        if Parameters.prediction_mode == "forecast_forward":
            _, sample = create_sample_prediction(Parameters.path_test_data)
        else:
            _, sample, _, _ = create_sample_gap_prediction(Parameters.path_test_data)
    else:
        if Parameters.prediction_mode == "forecast_forward":
            sample = sample.reshape((1, Parameters.lookback, 1))
        else:
            if Parameters.multiple_variables:
                sample = sample.reshape((1, Parameters.lookback + Parameters.length_of_prediction + Parameters.lookforward, Parameters.input_size))
            else:
                sample = sample.reshape((1, Parameters.lookback + Parameters.length_of_prediction + Parameters.lookforward, 1))

    # Create tensors from data arrays
    tensor_sample = torch.from_numpy(sample).float().to(device)

    print("Tensor shape:")
    print("Sample: " + str(sample.shape))

    # Predict based on input and send to numpy
    prediction = model(tensor_sample).cpu()
    prediction = prediction.detach().numpy()

    if Parameters.normalize_values:
        prediction = Denormalize(prediction, Parameters.data_max_value, Parameters.data_min_value)

    return prediction


def predict_iterative():
    current_series, _ = create_sample_prediction(Parameters.path_test_data)
    predicted_series = current_series.copy()

    for i in range(Parameters.number_of_predicts):
        predicted_series[Parameters.series_prediction_start + i] = predict(
            predicted_series[Parameters.series_prediction_start + i - Parameters.lookback:
                           Parameters.series_prediction_start + i])

        print(predicted_series[
              Parameters.series_prediction_start - Parameters.lookback + i:Parameters.series_prediction_start + i])

    for i in range(Parameters.series_prediction_start):
        predicted_series[i] = np.nan
    for i in range(Parameters.series_prediction_start + Parameters.number_of_predicts, len(predicted_series)):
        predicted_series[i] = np.nan

    return current_series, predicted_series


def predict_batch(start=Parameters.series_prediction_start):
    if Parameters.prediction_mode == "forecast_forward":
        current_series, sample = create_sample_prediction(Parameters.path_test_data)
        sample2 = 0
        sample3 = 0
    else:
        current_series, sample, sample2, sample3 = create_sample_gap_prediction(Parameters.path_test_data, start)

    predicted_series = current_series.copy()

    predicted_series = predicted_series.reshape((predicted_series.size, 1))

    if Parameters.prediction_mode == "forecast_forward":
        predicted_series[start:
                         start + Parameters.length_of_prediction] = predict(
            predicted_series[start - Parameters.lookback:
                             start])
    else:
        predicted_series[start:start + Parameters.length_of_prediction] = predict(sample)

    return current_series, predicted_series, sample2, sample3