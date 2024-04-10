import math

import matplotlib as mpl
import matplotlib.pyplot as plt

import Parameters

mpl.use('TkAgg')

def plot_data(original_data, predicted_data, start=Parameters.series_prediction_start):
    plt.plot(original_data, c='b', label="True data")
    predicted_data[:start] = math.nan
    predicted_data[start + Parameters.length_of_prediction:] = math.nan
    plt.plot(predicted_data, c='r', label="Prediction")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()