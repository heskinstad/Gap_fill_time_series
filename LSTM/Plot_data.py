import math

import matplotlib as mpl
import matplotlib.pyplot as plt

from LSTM import Parameters

mpl.use('TkAgg')

def plot_data(original_data, predicted_data):
    plt.plot(original_data, c='b')
    predicted_data[:Parameters.series_prediction_start] = math.nan
    predicted_data[Parameters.series_prediction_start + Parameters.length_of_prediction:] = math.nan
    plt.plot(predicted_data, c='r')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()