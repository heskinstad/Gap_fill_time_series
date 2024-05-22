import math
import random

import numpy as np
from matplotlib import pyplot as plt

import time

from scipy.stats import pearsonr

import LSTM.Process_csv
import Parameters
import Statistical_methods.Confidence_interval
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
        original_data, prediction, sample2, sample3 = predict_batch()
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

    plot_data(original_data, prediction, sample2=sample2)

elif Parameters.mode == "accuracy":
    number_of_tests = Parameters.number_of_tests

    data_len = len(LSTM.Process_csv.process_csv_column(Parameters.path_test_data, Parameters.column_index))

    if Parameters.accuracy_tests_from_array:
        mse_array = np.empty(len(Parameters.test_positions))
        mae_array = np.empty(len(Parameters.test_positions))
        corr_coeff_array = np.empty(len(Parameters.test_positions))
    else:
        mse_array = np.empty(number_of_tests)
        mae_array = np.empty(number_of_tests)
        corr_coeff_array = np.empty(number_of_tests)

    time_start = time.time()

    def run_accuracy_test(start):
        if Parameters.test_type == "LSTM":
            original_data, prediction, sample2, sample3 = predict_batch(start)

            original_data_crop = original_data.copy()[
                            start:start + Parameters.length_of_prediction]
            prediction_crop = prediction.copy()[
                         start:start + Parameters.length_of_prediction]

            mse = mean_squared_error(original_data_crop, prediction_crop)
            mae = mean_absolute_error(original_data_crop, prediction_crop)
            corr_coeff = pearsonr(original_data_crop, prediction_crop.flatten())[0]

            mse_array[test] = mse
            mae_array[test] = mae
            corr_coeff_array[test] = corr_coeff

            if Parameters.error_every_test:
                print("RNN LSTM Mean squared error: %.3f" % mse)
                print("RNN LSTM Mean absolute error: %.3f" % mae)
                print("RNN LSTM Correlation Coefficient: %.3f" % corr_coeff)

            if Parameters.plot_every_test:
                plot_data(original_data, prediction_crop.flatten(), start, sample2=sample2)

        elif Parameters.test_type == "ARIMA":
            mse_array[test], mae_array[test], corr_coeff_array[test] = run_ARIMA(start)

        elif Parameters.test_type == "interpolation":
            mse_array[test], mae_array[test], corr_coeff_array[test] = run_linear_interpolation(start)

    if Parameters.accuracy_tests_from_array:
        for test in range(len(Parameters.test_positions)):
            run_accuracy_test(Parameters.test_positions[test])
    else:
        for test in range(number_of_tests):
            start = random.randint(Parameters.lookback, data_len - Parameters.lookback - Parameters.length_of_prediction - 1)
            run_accuracy_test(start)

    mse = 0.0
    mae = 0.0
    corr_coeff = 0.0
    abs_corr_coeff = 0.0

    for i in range(len(corr_coeff_array)):
        if math.isnan(corr_coeff_array[i]):
            corr_coeff_array[i] = corr_coeff_array[i-1]  # Use the nearest value. These are pretty much the same and happens rarely, which will not affect the end results to any significant extent

    for i in range(len(mse_array)):
        mse += mse_array[i]
        mae += mae_array[i]
        corr_coeff += corr_coeff_array[i]
        abs_corr_coeff += abs(corr_coeff_array[i])

    if Parameters.accuracy_tests_from_array:
        mse /= len(Parameters.test_positions)
        mae /= len(Parameters.test_positions)
        corr_coeff /= len(Parameters.test_positions)
        abs_corr_coeff /= len(Parameters.test_positions)
    else:
        mse /= number_of_tests
        mae /= number_of_tests
        corr_coeff /= number_of_tests
        abs_corr_coeff /= number_of_tests

    print("Average mean squared error after %d runs: %.3f" % (number_of_tests, mse))
    print("Average mean absolute error after %d runs: %.3f" % (number_of_tests, mae))
    print("Average correlation coefficient after %d runs: %.3f" % (number_of_tests, corr_coeff))
    print("Average absolute correlation coefficient after %d runs: %.3f" % (number_of_tests, abs_corr_coeff))

    print("Time elapsed: %.3f seconds" % (time.time() - time_start))

else:
    current_series, sample, _ = create_sample_gap_prediction(Parameters.path_test_data)
    print(sample)
    plt.plot(sample)
    plt.show()

