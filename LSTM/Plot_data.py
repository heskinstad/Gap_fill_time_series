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
    original_data = original_data.copy()
    predicted_data[:start-1] = math.nan
    predicted_data[start + Parameters.length_of_prediction + 1:] = math.nan
    missing_data = original_data.copy()
    missing_data[:start - 1] = math.nan
    missing_data[start + Parameters.length_of_prediction + 1:] = math.nan
    missing_data = missing_data[start - Parameters.lookback - 100:start + Parameters.length_of_prediction + Parameters.lookforward + 100]
    original_data[start:start + Parameters.length_of_prediction] = math.nan
    original_data = original_data[start - Parameters.lookback - 100:start + Parameters.length_of_prediction + Parameters.lookforward + 100]
    predicted_data = predicted_data[start - Parameters.lookback - 100:start + Parameters.length_of_prediction + Parameters.lookforward + 100]

    dates = pd.to_datetime(LSTM.Process_csv.process_csv_column(Parameters.path_test_data, 1, has_header=True, datetimes=True)[start - Parameters.lookback - 100:start + Parameters.length_of_prediction + Parameters.lookforward + 100])

    fig, ax = plt.subplots()
    ax.plot(dates, original_data, c='b', label="True data")
    ax.plot(dates, predicted_data, c='r', label="Prediction")
    ax.plot(dates, missing_data, '--', c='b', label="True data (missing)")

    '''if Parameters.multiple_variables:
        sample2_array = np.empty(100+Parameters.lookback+Parameters.lookforward+Parameters.length_of_prediction)
        sample2_array[:] = math.nan
        sample2_array[100+Parameters.lookback:100+Parameters.lookback+Parameters.length_of_prediction] = sample2[10:-10]
        ax.plot(dates, sample2_array, '--', c='y', label="True data (2nd dataset)")'''

    ax.axvline(dates[100], ls='--', color='darkgray')
    ax.axvline(dates[100 + Parameters.lookback + Parameters.length_of_prediction + Parameters.lookforward], ls='--', color='darkgray')
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature")
    plt.legend()
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(MaxNLocator(10))  # Limit the number of x-axis ticks
    plt.gcf().autofmt_xdate()  # Auto rotates dates for better readability

    plt.show()