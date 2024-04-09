# Additional imports are required
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Plots
import matplotlib.pyplot as plt
from pylab import rcParams

from LSTM import Create_sample_target

rcParams['figure.figsize'] = 18, 7

import matplotlib as mpl
mpl.use('TkAgg')

import warnings
warnings.filterwarnings('ignore')

# Prerocessing for FEDOT
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

# FEDOT
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode

import logging
logging.raiseExceptions = False


def generate_synthetic_data(length: int = 2000, periods: int = 10):
    """
    The function generates a synthetic univariate time series

    :param length: the length of the array (even number)
    :param periods: the number of periods in the sine wave

    :return synthetic_data: an array without gaps
    """

    # First component
    sinusoidal_data = np.linspace(-periods * np.pi, periods * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)

    # Second component
    cos_1_data = np.linspace(-periods * np.pi / 2, periods / 2 * np.pi / 2, int(length / 2))
    cos_1_data = np.cos(cos_1_data)
    cos_2_data = np.linspace(periods / 2 * np.pi / 2, periods * np.pi / 2, int(length / 2))
    cos_2_data = np.cos(cos_2_data)
    cosine_data = np.hstack((cos_1_data, cos_2_data))

    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave, cos wave and random noise
    synthetic_data = sinusoidal_data + cosine_data + random_noise
    return synthetic_data


# Get such numpy array
synthetic_time_series = generate_synthetic_data()

# We will predict 100 values in the future
len_forecast = 100

synthetic_time_series, _ = Create_sample_target.create_sample_gap_prediction(os.getcwd() + r"\data\Munkholmen\all_hourly_fixed.csv")[:2000]

# Let's dividide our data on train and test samples
train_data = synthetic_time_series[:-len_forecast]
test_data = synthetic_time_series[-len_forecast:]

# Plot time series
plt.plot(np.arange(0, len(synthetic_time_series)), synthetic_time_series, label='Test')
plt.plot(np.arange(0, len(train_data)), train_data, label='Train')
plt.ylabel('Parameter', fontsize=15)
plt.xlabel('Time index', fontsize=15)
plt.legend(fontsize=15)
plt.title('Synthetic time series')
plt.grid()
plt.show()


def make_forecast(train_data, len_forecast: int, window_size: int, final_model: str = 'ridge'):
    """
    Function for predicting values in a time series

    :param train_data: one-dimensional numpy array to train pipeline
    :param len_forecast: amount of values for predictions
    :param window_size: moving window size
    :param final_model: model in the root node

    :return predicted_values: numpy array, forecast of model
    """

    # Here we define which task should we use, here we also define two main forecast length
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Prepare input data for prediction part
    start_forecast = len(train_data)
    end_forecast = start_forecast + len_forecast
    # Create forecast indices
    forecast_idx = np.arange(start_forecast, end_forecast)
    predict_input = InputData(idx=forecast_idx,
                              features=train_data,
                              target=test_data,
                              task=task,
                              data_type=DataTypesEnum.ts)

    # Create a pipeline "lagged -> <final_model>"
    node_lagged = PrimaryNode('lagged')

    # Define parameters to certain node
    node_lagged.parameters = {'window_size': window_size}
    node_ridge = SecondaryNode(final_model, nodes_from=[node_lagged])
    ridge_pipeline = Pipeline(node_ridge)

    # Fit pipeline
    ridge_pipeline.fit(train_input)

    # Predict. Pipeline return OutputData object
    predicted_output = ridge_pipeline.predict(predict_input)

    # Convert forecasted values into one-dimensional array
    forecast = np.ravel(np.array(predicted_output.predict))

    return forecast


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
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()

predicted_values = make_forecast(train_data = train_data,
                                 len_forecast = 100,
                                 window_size = 400)

plot_results(actual_time_series = synthetic_time_series,
             predicted_values = predicted_values,
             len_train_data = len(train_data))