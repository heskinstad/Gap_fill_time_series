import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib as mpl
import Parameters
from LSTM.Create_sample_target import create_sample_target_ARIMA
import numpy as np

mpl.use('TkAgg')

def run_ARIMA(start=Parameters.series_prediction_start, show_plot=True):

    # Retrieve sample and target from file
    sample_before, target, sample_after = create_sample_target_ARIMA(Parameters.path_test_data, start)

    target_repositioned = np.empty(len(sample_before) + len(target), dtype=float)
    target_repositioned[:] = np.nan
    target_repositioned[len(sample_before):] = target

    sample_between = np.empty(len(target), dtype=float)
    sample_between[:] = np.nan
    sample_combined = np.concatenate((sample_before, sample_between, sample_after))

    index_sample_before = pd.date_range(start='2024-01-01', periods=len(sample_before), freq='H')  # One point every hour
    index_target = pd.date_range(start='2024-01-01', periods=len(target_repositioned), freq='H')  # One point every hour
    index_sample_after = pd.date_range(start='2024-01-01', periods=len(sample_after), freq='H')  # One point every hour
    index_sample_combined = pd.date_range(start='2024-01-01', periods=len(sample_combined), freq='H')  # One point every hour

    sample_series_before = pd.Series(sample_before, index_sample_before)
    target_series = pd.Series(target_repositioned, index_target)
    sample_series_after = pd.Series(np.flip(sample_after), index_sample_after)  # Flip sample_after because array must be reversed
    sample_series_combined = pd.Series(sample_combined, index_sample_combined)


    # Fit the ARIMA model - ARIMA(p, d, q)
    model_forward = ARIMA(sample_series_before, order=(Parameters.p, Parameters.d, Parameters.q))
    model_fit_forward = model_forward.fit()
    model_backward = ARIMA(sample_series_after, order=(Parameters.p, Parameters.d, Parameters.q))
    model_fit_backward = model_backward.fit()

    # Forecast the next n steps
    forecast_forward = model_fit_forward.forecast(steps=Parameters.length_of_prediction)
    forecast_backward = np.flip(model_fit_backward.forecast(steps=Parameters.length_of_prediction))

    # Get weighted average element-wise from the two forecast arrays
    weights = np.arange(0, 1, 1.0/Parameters.length_of_prediction, dtype=float)
    forecast_weighted_average = np.empty(Parameters.length_of_prediction, dtype=float)
    for i in range(len(forecast_weighted_average)):
        forecast_weighted_average[i] = forecast_forward[i] * abs(weights[i] - 1) + forecast_backward[i] * weights[i]

    # Generate the date range for the forecast
    forecast_dates_forward = pd.date_range(start=index_sample_before[-1], periods=Parameters.length_of_prediction + 1, freq='H')[1:]
    forecast_dates_backward = pd.date_range(start=index_sample_after[-1], periods=Parameters.length_of_prediction + 1, freq='H')[1:]

    # Prepare the target for plotting
    target_repositioned = np.empty(len(sample_before)+len(target))
    target_dates = pd.date_range(start='2024-01-01', periods=len(target_repositioned) + 1, freq='H')[1:]

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(target_series[Parameters.lookback:], forecast_weighted_average)
    mae = mean_absolute_error(target_series[Parameters.lookback:], forecast_weighted_average)

    if show_plot:
        # Plot the historical data and future predictions
        plt.figure(figsize=(10, 6))
        plt.plot(sample_series_combined, label='Historical Data')
        plt.plot(forecast_dates_forward, forecast_forward, '--', label='ARIMA Forecast forward')
        plt.plot(forecast_dates_backward, forecast_backward, '--', label='ARIMA Forecast backward')
        plt.plot(forecast_dates_forward, forecast_weighted_average, label='ARIMA Forecast weighted average mean')
        plt.plot(target_series, label='True Data')
        plt.legend()
        plt.show()

        print("ARIMA Mean squared error: %.3f" % mse)
        print("ARIMA Mean absolute error: %.3f" % mae)

    return mse, mae

#run_ARIMA()