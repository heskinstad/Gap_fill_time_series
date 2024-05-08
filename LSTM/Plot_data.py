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

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(dates, original_data, c='b', label="True data")
    ax.plot(dates[values_before+Parameters.lookback:values_before+Parameters.lookback+Parameters.length_of_prediction], predicted_data, c='r', label="Prediction")
    ax.plot(dates, missing_data, '--', c='b', label="True data (missing)")


    # Add multivariate time-series
    if Parameters.multiple_variables and Parameters.test_type == "LSTM":
        sample2_array = np.empty(Parameters.lookback+Parameters.lookforward+Parameters.length_of_prediction+10)
        sample2_array[:] = math.nan
        sample2_array[values_before+Parameters.lookback:values_before+Parameters.lookback+Parameters.length_of_prediction] = sample2[Parameters.lookback:-Parameters.lookforward]
        ax.plot(dates[:Parameters.lookback+Parameters.lookforward+Parameters.length_of_prediction+10], sample2_array, '--', c='g', label="Available data (MET)")

    if Parameters.test_type == "LSTM":
        ax.axvspan(dates[values_before], dates[values_before+Parameters.lookback], facecolor='green', alpha=0.2, label="Available data")
        ax.axvspan(dates[values_before + Parameters.lookback + Parameters.length_of_prediction], dates[values_before + Parameters.lookback + Parameters.length_of_prediction + Parameters.lookforward], facecolor='green', alpha=0.2)

    plt.xlabel("Time (date)")
    plt.ylabel("Temperature (°C)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, fancybox=True, shadow=True)
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(MaxNLocator(10))  # Limit the number of x-axis ticks
    plt.gcf().autofmt_xdate()  # Auto rotates dates for better readability

    plt.show()