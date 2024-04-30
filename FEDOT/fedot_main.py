# Additional imports
import os

import pandas as pd
import numpy as np

# Imports for creating plots
import matplotlib.pyplot as plt
from pylab import rcParams

import Parameters
from LSTM import Create_sample_target

rcParams['figure.figsize'] = 18, 7

# Pipeline and nodes
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode

# Data
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum

# Tasks
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

import logging
logging.raiseExceptions = False

import matplotlib as mpl
mpl.use('TkAgg')

# Read the file
df, _, _ = Create_sample_target.create_sample_gap_prediction(os.getcwd() + r"\data\Munkholmen\all_hourly_fixed.csv")

#df = pd.to_datetime(df)
print(df)

plt.plot(df)
plt.show()


# Let's prepare a function for visualizing forecasts
def plot_results(actual_time_series, predicted_values, len_train_data, y_name='Parameter'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black', linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15, loc='upper left')
    plt.grid()
    plt.show()


# Specify forecast length
forecast_length = Parameters.length_of_prediction

# Got univariate time series as numpy array
time_series = df

# Wrapp data into InputData
task = Task(TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length=forecast_length))
input_data = InputData(idx=np.arange(0, len(time_series)),
                       features=time_series,
                       target=time_series,
                       task=task,
                       data_type=DataTypesEnum.ts)

# Split data into train and test
train_input, predict_input = train_test_data_setup(input_data)


def get_pipeline():
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_1.parameters = {'window_size': 50}
    node_lagged_2 = PrimaryNode('lagged')
    node_lagged_2.parameters = {'window_size': 10}

    node_first = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_second = SecondaryNode('dtreg', nodes_from=[node_lagged_2])
    node_final = SecondaryNode('ridge', nodes_from=[node_first, node_second])
    pipeline = Pipeline(node_final)

    return pipeline

pipeline = get_pipeline()


# Fit pipeline
pipeline.fit(train_input)

# Make forecast
output = pipeline.predict(predict_input)
forecast = np.ravel(np.array(output.predict))

# Plot the graph
plot_results(actual_time_series = time_series,
             predicted_values = forecast,
             len_train_data = len(time_series)-forecast_length)