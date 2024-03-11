import datetime

import pandas as pd
from cartopy import mpl
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib as mpl
import Parameters
from LSTM.Create_sample_target import create_sample_target_ARIMA
import numpy as np

mpl.use('TkAgg')


# Retrieve sample and target from file
sample, target = create_sample_target_ARIMA(Parameters.path_test_data)
target_repositioned = np.empty(len(sample) + len(target), dtype=float)
target_repositioned[:] = np.nan
target_repositioned[len(target):] = target
index_sample = pd.date_range(start='2020-01-01', periods=len(sample), freq='H')  # One point every hour
index_target = pd.date_range(start='2020-01-01', periods=len(target_repositioned), freq='H')  # One point every hour
sample_series = pd.Series(sample, index_sample)
target_series = pd.Series(target_repositioned, index_target)

# Fit the ARIMA model (with p=1, d=1, q=1 as an example)
model = ARIMA(sample_series, order=(50, 0, 4))
model_fit = model.fit()

# Forecast the next 5 steps
forecast = model_fit.forecast(steps=Parameters.lookback)
print(forecast)

# Generate the date range for the forecast
forecast_dates = pd.date_range(start=index_sample[-1], periods=Parameters.length_of_prediction + 1, freq='H')[1:]

# Prepare the target for plotting
target_repositioned = np.empty(sample.size+target.size)
target_dates = pd.date_range(start='2020-01-01', periods=len(target_repositioned) + 1, freq='H')[1:]

# Plot the historical data and future predictions
plt.figure(figsize=(10, 6))
plt.plot(sample_series, label='Historical Data')
plt.plot(forecast_dates, forecast, label='ARIMA Forecast')
plt.plot(target_series, label='True Data')
plt.legend()
plt.show()