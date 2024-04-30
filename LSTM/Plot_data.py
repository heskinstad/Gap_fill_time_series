import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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


    plt.figure(figsize=(10,6))
    plt.plot(original_data, c='b', label="True data")
    plt.plot(predicted_data, c='r', label="Prediction")
    plt.plot(missing_data, '--', c='b', label="True data (missing)")

    if Parameters.multiple_variables:
        sample2_array = np.empty(100+Parameters.lookback+Parameters.lookforward+Parameters.length_of_prediction)
        sample2_array[:] = math.nan
        sample2_array[100+Parameters.lookback:100+Parameters.lookback+Parameters.length_of_prediction] = sample2[10:-10]
        plt.plot(sample2_array, '--', c='y', label="True data (2nd dataset)")

    plt.axvline(100, ls='--', color='darkgray')
    plt.axvline(100 + Parameters.lookback + Parameters.length_of_prediction + Parameters.lookforward, ls='--', color='darkgray')
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()