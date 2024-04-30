import random
import numpy as np

import Parameters
from LSTM.Process_csv import process_csv_row, process_csv_column
from LSTM.Normalize import Normalize


def create_sample_target_training(path):
    num_of_sample_targets_per_series = Parameters.num_of_sample_targets_per_series
    total_num_of_series = Parameters.total_num_of_series

    # Number of measurements to use for prediction
    lookback = Parameters.lookback

    samples = np.empty((total_num_of_series * num_of_sample_targets_per_series, lookback, 1))
    targets = np.empty((total_num_of_series * num_of_sample_targets_per_series, Parameters.length_of_prediction, 1))

    for i in range(total_num_of_series):
        if Parameters.column_or_row == "row":
            current_series = np.array(process_csv_row(path, i + Parameters.row_index), dtype=float)
        else:
            current_series = np.array(process_csv_column(path, i + Parameters.column_index), dtype=float)

        series_length = current_series.size

        for j in range(num_of_sample_targets_per_series):
            # Create n number of sample-targets from this one series
            start = random.randint(0, series_length - lookback - Parameters.length_of_prediction - 1)
            sample = current_series[start:start+lookback]
            target = current_series[start+lookback:start + lookback + Parameters.length_of_prediction]

            sample = np.expand_dims(sample, axis=1)
            target = np.expand_dims(target, axis=1)

            samples[i*j + j] = sample
            targets[i*j + j] = target

    return samples, targets


def create_sample_prediction(path):
    if Parameters.column_or_row == "row":
        current_series = np.array(process_csv_row(path, Parameters.prediction_series_row), dtype=float)
    else:
        current_series = np.array(process_csv_column(path, Parameters.prediction_series_column), dtype=float)

    sample = current_series[Parameters.series_prediction_start - Parameters.lookback:Parameters.series_prediction_start]

    sample = sample.reshape(1, Parameters.lookback, 1)

    return current_series, sample


def create_sample_target_gap_training(path, path2):
    num_of_sample_targets_per_series = Parameters.num_of_sample_targets_per_series
    total_num_of_series = Parameters.total_num_of_series

    # Number of measurements to use for prediction
    lookback = Parameters.lookback
    length_of_prediction = Parameters.length_of_prediction
    lookforward = Parameters.lookforward

    if Parameters.multiple_variables:
        samples = np.empty((total_num_of_series * num_of_sample_targets_per_series, lookback + length_of_prediction + lookforward, 2))
    else:
        samples = np.empty((total_num_of_series * num_of_sample_targets_per_series, lookback + length_of_prediction + lookforward, 1))

    targets = np.empty((total_num_of_series * num_of_sample_targets_per_series, length_of_prediction, 1))

    for i in range(total_num_of_series):
        if Parameters.column_or_row == "row":
            current_series = np.array(process_csv_row(path, i + Parameters.row_index), dtype=float)
        else:
            current_series = np.array(process_csv_column(path, i + Parameters.column_index), dtype=float)
            if Parameters.multiple_variables:
                current_series_2 = np.array(process_csv_column(path2, i + Parameters.column_index_second_variable), dtype=float)

        series_length = current_series.size

        for j in range(num_of_sample_targets_per_series):
            # Create n number of sample-targets from this one series
            start = random.randint(0, series_length - lookback - length_of_prediction - lookforward - 1)

            sample = current_series.copy()[start:start + lookback + Parameters.length_of_prediction + lookforward]
            if Parameters.multiple_variables:
                sample2 = current_series_2.copy()[start:start + lookback + Parameters.length_of_prediction + lookforward]
                sample2[:lookback] = 0
                sample2[lookback+length_of_prediction:] = 0

            sample[lookback:lookback+length_of_prediction] = 0

            target = current_series.copy()[start + lookback:start + lookback + Parameters.length_of_prediction]

            if Parameters.multiple_variables:
                sample = np.column_stack((sample, sample2))

            if not Parameters.multiple_variables:
                sample = np.expand_dims(sample, axis=1)
            target = np.expand_dims(target, axis=1)

            samples[i*j + j] = sample
            targets[i*j + j] = target

        if Parameters.normalize_values:
            samples = Normalize(samples, Parameters.data_max_value, Parameters.data_min_value)
            targets = Normalize(targets, Parameters.data_max_value, Parameters.data_min_value)

    return samples, targets


def create_sample_gap_prediction(path, start=Parameters.series_prediction_start):
    if Parameters.column_or_row == "row":
        current_series = np.array(process_csv_row(path, Parameters.prediction_series_row), dtype=float)
    else:
        current_series = np.array(process_csv_column(path, Parameters.prediction_series_column), dtype=float)
        if Parameters.multiple_variables:
            current_series_2 = np.array(process_csv_column(Parameters.path_test_data_other_variable, Parameters.prediction_series_column), dtype=float)

    sample = current_series.copy()[
             start - Parameters.lookback:start + Parameters.length_of_prediction + Parameters.lookforward]
    if Parameters.multiple_variables:
        sample2 = current_series_2.copy()[start - Parameters.lookback:start + Parameters.length_of_prediction + Parameters.lookforward]
        sample2[:Parameters.lookback] = 0
        sample2[Parameters.lookback + Parameters.length_of_prediction:] = 0

    sample[Parameters.lookback:Parameters.lookback + Parameters.length_of_prediction] = 0

    if Parameters.multiple_variables:
        sample = np.column_stack((sample, sample2))

    if Parameters.normalize_values:
        sample = Normalize(sample, Parameters.data_max_value, Parameters.data_min_value)

    if not Parameters.multiple_variables:
        sample2 = 0

    return current_series, sample, sample2


def create_sample_target_ARIMA(path, start=Parameters.series_prediction_start):
    if Parameters.column_or_row == "row":
        current_series = np.array(process_csv_row(path, Parameters.prediction_series_row), dtype=float)
    else:
        current_series = np.array(process_csv_column(path, Parameters.prediction_series_column), dtype=float)

    sample_before = current_series[
                    start - Parameters.lookback:start]
    target = current_series[
             start:start + Parameters.length_of_prediction]
    sample_after = current_series[
                   start + Parameters.length_of_prediction:start + Parameters.length_of_prediction + Parameters.lookforward]

    return sample_before, target, sample_after