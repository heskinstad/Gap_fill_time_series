import random
import numpy as np

import Parameters
from Process_csv import process_csv_row, process_csv_column


def create_sample_target_training(path):
    num_of_sample_targets_per_series = Parameters.num_of_sample_targets_per_series
    total_num_of_series = Parameters.total_num_of_series

    # Number of measurements to use for prediction
    lookback = Parameters.lookback

    samples = np.empty((total_num_of_series * num_of_sample_targets_per_series, lookback))
    targets = np.empty((total_num_of_series * num_of_sample_targets_per_series, 1))

    for i in range(total_num_of_series):
        current_series = np.array(process_csv_row(path, i+1), dtype=float)
        series_length = current_series.size

        for j in range(num_of_sample_targets_per_series):
            # Create n number of sample-targets from this one series
            start = random.randint(0, series_length-lookback-2)
            sample = current_series[start:start+lookback]
            target = current_series[start+lookback+1]

            samples[i*j + j] = sample
            targets[i*j + j] = target

    return samples, targets


def create_sample_prediction(path):
    current_series = np.array(process_csv_row(path, Parameters.prediction_series), dtype=float)

    sample = np.empty((1, Parameters.lookback), dtype=float)

    sample[0] = current_series[Parameters.series_prediction_start-Parameters.lookback:Parameters.series_prediction_start]

    return current_series, sample