import random

import numpy as np
from matplotlib import pyplot as plt

import Parameters
from LSTM.Create_sample_target import create_sample_target_gap_training, create_sample_gap_prediction
from Train import train_model
from Predict import predict_iterative, predict_batch
from Plot_data import plot_data

if Parameters.mode == "train":
    train_model()

elif Parameters.mode == "predict":
    if Parameters.length_of_prediction > 1:
        original_data, prediction = predict_batch()
    else:
        original_data, prediction = predict_iterative()

    ### MEAN SQUARED/ABSOLUTE ERROR ###
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print("Mean squared error: %.3f" % mean_squared_error(
        prediction[Parameters.series_prediction_start:
        Parameters.series_prediction_start+Parameters.length_of_prediction],
        original_data[Parameters.series_prediction_start:
        Parameters.series_prediction_start+Parameters.length_of_prediction]))
    print("Mean absolute error: %.3f" % mean_absolute_error(
        prediction[Parameters.series_prediction_start:
                   Parameters.series_prediction_start + Parameters.length_of_prediction],
        original_data[Parameters.series_prediction_start:
                      Parameters.series_prediction_start + Parameters.length_of_prediction]))
    ##########################

    plot_data(original_data, prediction)

elif Parameters.mode == "test_accuracy":
    accuracy_array = np.empty(10)

    data_len, _ = predict_batch()
    data_len = data_len.size()

    for i in range(10):
        start = random.randint(0, data_len - Parameters.lookback - Parameters.length_of_prediction - 1 - Parameters.lookforward)
        original_data, prediction = predict_batch(start)
        original_data = original_data[
                        Parameters.series_prediction_start:Parameters.series_prediction_start + Parameters.lookforward]
        prediction = prediction[
                     Parameters.series_prediction_start:Parameters.series_prediction_start + Parameters.lookforward]

        #accuracy_array[i] = mean_squared_error()

elif Parameters.mode == "tete":
    samples, targets = create_sample_target_gap_training(Parameters.path_train_data)

    plt.plot(samples[375])
    plt.plot(targets[375])

    plt.show()

else:
    current_series, sample = create_sample_gap_prediction(Parameters.path_test_data)
    print(sample)
    plt.plot(sample)
    plt.show()

