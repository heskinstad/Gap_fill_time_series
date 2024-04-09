import random

import numpy as np
from matplotlib import pyplot as plt

import Parameters
from LSTM.Create_sample_target import create_sample_target_gap_training, create_sample_gap_prediction
from Train import train_model
from Predict import predict_iterative, predict_batch
from Plot_data import plot_data
from sklearn.metrics import mean_squared_error, mean_absolute_error

if Parameters.mode == "train":
    train_model()

elif Parameters.mode == "predict":
    if Parameters.length_of_prediction > 1:
        original_data, prediction = predict_batch()
    else:
        original_data, prediction = predict_iterative()

    ### MEAN SQUARED/ABSOLUTE ERROR ###
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

elif Parameters.mode == "accuracy":
    number_of_tests = Parameters.number_of_tests

    data_len, _ = predict_batch()
    data_len = len(data_len)

    mse_array = np.empty(number_of_tests)
    mae_array = np.empty(number_of_tests)

    for test in range(number_of_tests):
        start = random.randint(Parameters.lookback, data_len - Parameters.lookback - Parameters.length_of_prediction - 1)
        #print(start)
        original_data, prediction = predict_batch(start)
        #plot_data(original_data, prediction, start)
        original_data = original_data[
                        start:start + Parameters.length_of_prediction]
        prediction = prediction[
                     start:start + Parameters.length_of_prediction]

        mse_array[test] = mean_squared_error(original_data, prediction)
        mae_array[test] = mean_absolute_error(original_data, prediction)

    mse = 0.0
    mae = 0.0

    for i in range(len(mse_array)):
        mse += mse_array[i]
        mae += mae_array[i]

    mse /= number_of_tests
    mae /= number_of_tests

    #print(mse_array)

    print("Mean squared error: %.3f" % mse)
    print("Mean absolute error: %.3f" % mae)

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

