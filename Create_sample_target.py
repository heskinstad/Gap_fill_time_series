import random
import numpy as np
from sklearn import preprocessing

import Parameters
from Process_csv import process_csv_row, process_csv_column


def create_sample_target_training(path):
    num_of_sample_targets_per_series = Parameters.num_of_sample_targets_per_series
    total_num_of_series = Parameters.total_num_of_series

    # Number of measurements to use for prediction
    lookback = Parameters.lookback

    samples = np.empty((total_num_of_series * num_of_sample_targets_per_series, lookback, 1))
    targets = np.empty((total_num_of_series * num_of_sample_targets_per_series, 1, 1))

    for i in range(total_num_of_series):
        if Parameters.column_or_row == "row":
            current_series = np.array(process_csv_row(path, i+Parameters.row_index), dtype=float)
        else:
            current_series = np.array(process_csv_column(path, i + Parameters.column_index), dtype=float)

        series_length = current_series.size

        for j in range(num_of_sample_targets_per_series):
            # Create n number of sample-targets from this one series
            start = random.randint(0, series_length-lookback-2)
            sample = current_series[start:start+lookback]
            target = current_series[start+lookback:start+lookback+1]

            sample = np.expand_dims(sample, axis=1)
            target = np.expand_dims(target, axis=1)

            print(sample.shape)
            print(sample)
            print(target.shape)
            print(target)

            samples[i*j + j] = sample
            targets[i*j + j] = target

    return samples, targets


def create_sample_prediction(path):
    if Parameters.column_or_row == "row":
        current_series = np.array(process_csv_row(path, Parameters.prediction_series_row), dtype=float)
    else:
        current_series = np.array(process_csv_column(path, Parameters.prediction_series_column), dtype=float)

    sample = current_series[Parameters.series_prediction_start-Parameters.lookback:Parameters.series_prediction_start]

    sample = sample.reshape(1, Parameters.lookback, 1)

    return current_series, sample