import random

import numpy as np
from matplotlib import pyplot as plt

import time

import Parameters
from LSTM.Create_sample_target import create_sample_target_gap_training, create_sample_gap_prediction
from Statistical_methods.ARIMA import run_ARIMA
from Statistical_methods.Linear_interpolation import run_linear_interpolation
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
    print("RNN LSTM Mean squared error: %.3f" % mean_squared_error(
        prediction[Parameters.series_prediction_start:
        Parameters.series_prediction_start+Parameters.length_of_prediction],
        original_data[Parameters.series_prediction_start:
        Parameters.series_prediction_start+Parameters.length_of_prediction]))
    print("RNN LSTM Mean absolute error: %.3f" % mean_absolute_error(
        prediction[Parameters.series_prediction_start:
                   Parameters.series_prediction_start + Parameters.length_of_prediction],
        original_data[Parameters.series_prediction_start:
                      Parameters.series_prediction_start + Parameters.length_of_prediction]))
    ##########################

    plot_data(original_data, prediction)

elif Parameters.mode == "accuracy":
    number_of_tests = Parameters.number_of_tests

    data_len, _, _ = predict_batch()
    data_len = len(data_len)

    mse_array = np.empty(number_of_tests)
    mae_array = np.empty(number_of_tests)

    time_start = time.time()

    for test in range(number_of_tests):
        start = random.randint(Parameters.lookback, data_len - Parameters.lookback - Parameters.length_of_prediction - 1)

        if Parameters.test_type == "LSTM":
            original_data, prediction, sample2 = predict_batch(start)
            if Parameters.plot_every_test:
                plot_data(original_data, prediction, start, sample2=sample2)
            original_data = original_data[
                            start:start + Parameters.length_of_prediction]
            prediction = prediction[
                         start:start + Parameters.length_of_prediction]

            mse = mean_squared_error(original_data, prediction)
            mae = mean_absolute_error(original_data, prediction)

            mse_array[test] = mse
            mae_array[test] = mae

            if Parameters.error_every_test:
                print("RNN LSTM Mean squared error: %.3f" % mse)
                print("RNN LSTM Mean absolute error: %.3f" % mae)

        elif Parameters.test_type == "ARIMA":
            mse_array[test], mae_array[test] = run_ARIMA(start)

        elif Parameters.test_type == "interpolation":
            mse_array[test], mae_array[test] = run_linear_interpolation(start)

    mse = 0.0
    mae = 0.0

    for i in range(len(mse_array)):
        mse += mse_array[i]
        mae += mae_array[i]

    mse /= number_of_tests
    mae /= number_of_tests

    print("Average mean squared error after %d runs: %.3f" % (number_of_tests, mse))
    print("Average mean absolute error after %d runs: %.3f" % (number_of_tests, mae))

    print("Time elapsed: %.3f seconds" % (time.time() - time_start))

elif Parameters.mode == "tete":
    samples, targets = create_sample_target_gap_training(Parameters.path_train_data)

    plt.plot(samples[375])
    plt.plot(targets[375])

    plt.show()

else:
    current_series, sample, _ = create_sample_gap_prediction(Parameters.path_test_data)
    print(sample)
    plt.plot(sample)
    plt.show()

