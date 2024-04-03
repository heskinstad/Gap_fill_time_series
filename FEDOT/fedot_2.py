# Additional imports are required
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Plots
import matplotlib.pyplot as plt
from pylab import rcParams

import matplotlib as mpl
mpl.use('TkAgg')

import Parameters
from LSTM import Create_sample_target, Process_csv

rcParams['figure.figsize'] = 18, 7

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

df_train = Process_csv.process_csv_column_with_timestamps(Parameters.path_train_data)
df_test = Process_csv.process_csv_column_with_timestamps(Parameters.path_test_data)

df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])

#true_values = np.concatenate((train_array, test_array))

# Specify forecast length
len_forecast = 250

# Got train, test parts, and the entire data

plt.plot(df_test['timestamp'], df_test['temp'], label = 'Test')
plt.plot(df_train['timestamp'], df_train['temp'], label = 'Train')
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Sea level, m', fontsize = 15)
plt.legend(fontsize = 15)
plt.grid()
plt.show()


###################
# Simple pipeline #
###################

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
                              target=df_test,
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
    print(predict_input)
    predicted_output = ridge_pipeline.predict(predict_input)

    # Convert forecasted values into one-dimensional array
    forecast = np.ravel(np.array(predicted_output.predict))

    return forecast

# Make predictions
predicted_values = make_forecast(train_data=df_train['temp'],
                                 len_forecast=250,
                                 window_size=400,
                                 final_model='ridge')

# Plot predictions and true values
plot_results(actual_time_series=df_train,
             predicted_values=predicted_values,
             len_train_data=len(df_train),
             y_name='Sea level, m')

# Print MAE metric
print(f'Mean absolute error: {mean_absolute_error(df_test, predicted_values):.3f}')