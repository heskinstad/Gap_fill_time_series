import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

import LSTM.Process_csv
import Parameters

mpl.use('TkAgg')

def plot_data(original_data, predicted_data, start=Parameters.series_prediction_start, sample2=0):
    values_before = 10
    if start < Parameters.lookback+values_before:
        values_before = start - Parameters.lookback

    original_data = original_data.copy()
    missing_data = original_data.copy()
    missing_data[:start - 1] = math.nan
    missing_data[start + Parameters.length_of_prediction + 1:] = math.nan
    missing_data = missing_data[start - Parameters.lookback - values_before:start + Parameters.length_of_prediction + Parameters.lookforward + 10]
    original_data[start:start + Parameters.length_of_prediction] = math.nan
    original_data = original_data[start - Parameters.lookback - values_before:start + Parameters.length_of_prediction + Parameters.lookforward + 10]

    dates = pd.to_datetime(LSTM.Process_csv.process_csv_column(Parameters.path_test_data, 1, has_header=True, datetimes=True)[start - Parameters.lookback - values_before:start + Parameters.length_of_prediction + Parameters.lookforward + 10])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.grid()
    ax.plot(dates, original_data, c='b', label="True data")
    ax.plot(dates[values_before+Parameters.lookback:values_before+Parameters.lookback+Parameters.length_of_prediction], predicted_data, c='r', label="Prediction")
    ax.plot(dates, missing_data, '--', c='b', label="True data (missing)")

    # Calculate confidence intervals
    lower_bound = np.percentile(missing_data[10+Parameters.lookback:10+Parameters.lookback+Parameters.length_of_prediction], 2.5, axis=0)
    upper_bound = np.percentile(missing_data[10+Parameters.lookback:10+Parameters.lookback+Parameters.length_of_prediction], 97.5, axis=0)
    interval = (upper_bound - lower_bound) / 2

    # Add multivariate time-series
    if Parameters.multiple_variables and Parameters.test_type == "LSTM":
        sample2_array = np.empty(Parameters.lookback+Parameters.lookforward+Parameters.length_of_prediction+10)
        sample2_array[:] = math.nan
        sample2_array[values_before+Parameters.lookback:values_before+Parameters.lookback+Parameters.length_of_prediction] = sample2[Parameters.lookback:-Parameters.lookforward]
        ax.plot(dates[:Parameters.lookback+Parameters.lookforward+Parameters.length_of_prediction+10], sample2_array, '--', c='cyan', label="Available data (MET)")

    if Parameters.test_type == "LSTM":
        ax.axvspan(dates[values_before], dates[values_before+Parameters.lookback-1], facecolor='green', alpha=0.2, label="Available data")
        ax.axvspan(dates[values_before + Parameters.lookback + Parameters.length_of_prediction], dates[values_before + Parameters.lookback + Parameters.length_of_prediction + Parameters.lookforward], facecolor='green', alpha=0.2)

    # Plot confidence interval
    plt.fill_between(dates[10+Parameters.lookback:10+Parameters.lookback+Parameters.length_of_prediction],
                     missing_data[10+Parameters.lookback:10+Parameters.lookback+Parameters.length_of_prediction]-interval,
                     missing_data[10+Parameters.lookback:10+Parameters.lookback+Parameters.length_of_prediction]+interval, color='blue', alpha=0.1)

    plt.xlabel("Time\n(date)")
    ax.xaxis.set_label_coords(-0.10, -0.038)
    plt.ylabel("Temperature (°C)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), ncol=3)
    if Parameters.test_type == "LSTM":
        plt.title("RNN LSTM gap-fill")
    elif Parameters.test_type == "interpolation":
        plt.title("Linear interpolation gap-fill")
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.12)
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m-%d'))
    plt.gca().xaxis.set_major_locator(MaxNLocator(10))  # Limit the number of x-axis ticks
    plt.gcf().autofmt_xdate()  # Auto rotates dates for better readability

    plt.show()