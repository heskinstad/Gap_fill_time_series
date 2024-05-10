import matplotlib
import pandas as pd
from matplotlib import ticker
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import Parameters
from LSTM.Create_sample_target import create_sample_target_ARIMA
from LSTM.Process_csv import process_csv_column
import numpy as np

mpl.use('TkAgg')

def run_ARIMA(start=Parameters.series_prediction_start):

    start = start-Parameters.lookback  # Small error in code with setting the date. Easy solution

    values_before = 10
    if start < Parameters.lookback - 10:
        values_before = start - Parameters.lookback
    dates = pd.to_datetime(process_csv_column(Parameters.path_test_data, 1, has_header=True, datetimes=True))

    # Retrieve sample and target from file
    sample_before, target, sample_after = create_sample_target_ARIMA(Parameters.path_test_data, start+Parameters.lookback)

    target_repositioned = np.empty(len(sample_before) + len(target), dtype=float)
    target_repositioned[:] = np.nan
    target_repositioned[len(sample_before):] = target

    sample_between = np.empty(len(target), dtype=float)
    sample_between[:] = np.nan
    sample_combined = np.concatenate((sample_before, sample_between, sample_after))

    index_sample_before = pd.date_range(start=dates[start], periods=len(sample_before), freq='H')  # One point every hour
    index_target = pd.date_range(start=dates[start], periods=len(target_repositioned), freq='H')  # One point every hour
    index_sample_after = pd.date_range(start=dates[start], periods=len(sample_after), freq='H')  # One point every hour
    index_sample_combined = pd.date_range(start=dates[start], periods=len(sample_combined), freq='H')  # One point every hour

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

    mse = mean_squared_error(target_series[Parameters.lookback:], forecast_weighted_average)
    mae = mean_absolute_error(target_series[Parameters.lookback:], forecast_weighted_average)
    corr_coeff = pearsonr(target_series[Parameters.lookback:], forecast_weighted_average)[0]

    data_before, _, _ = create_sample_target_ARIMA(Parameters.path_test_data, start+1)
    _, _, data_after = create_sample_target_ARIMA(Parameters.path_test_data, start+Parameters.lookback+Parameters.lookforward-1)

    if Parameters.error_every_test:
        print("ARIMA Mean squared error: %.3f" % mse)
        print("ARIMA Mean absolute error: %.3f" % mae)
        print("Correlation Coefficient error: %.3f" % corr_coeff)

    if Parameters.plot_every_test:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.grid()
        ax.plot(sample_series_combined, label='True Data', color='b')
        ax.plot(dates[start-values_before+1:start+1], data_before[Parameters.lookback-values_before:Parameters.lookback], color='b')
        ax.plot(dates[start+Parameters.lookback+Parameters.length_of_prediction+Parameters.lookforward-1:start+Parameters.lookback+Parameters.length_of_prediction+Parameters.lookforward+9], data_after[:10], color='b')
        ax.plot(target_series, '--', label='True Data (missing)', color='b')
        ax.plot(forecast_dates_forward, forecast_weighted_average, label='Forecast weighted average mean', color='r')
        ax.plot(forecast_dates_forward, forecast_forward, '--', label='Forecast forward', color='g', alpha=0.5)
        ax.plot(forecast_dates_backward, forecast_backward, '--', label='Forecast backward', color='y', alpha=0.5)
        ax.axvspan(dates[start], dates[start+Parameters.lookback], facecolor='green', alpha=0.2,
                   label="Available data pre-gap")
        ax.axvspan(dates[start+Parameters.lookback+Parameters.length_of_prediction], dates[
            start+Parameters.lookback+Parameters.length_of_prediction+Parameters.lookforward],
                   facecolor='yellow', alpha=0.2, label="Available data post-gap")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), ncol=3)
        plt.xlabel("Time\n(date)")
        ax.xaxis.set_label_coords(-0.1, -0.038)
        plt.ylabel("Temperature (°C)")
        plt.title("ARIMA gap-fill")
        plt.subplots_adjust(bottom=0.2)
        plt.subplots_adjust(left=0.12)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m-%d'))
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(10))  # Limit the number of x-axis ticks
        plt.gcf().autofmt_xdate()  # Auto rotates dates for better readability
        plt.show()

    return mse, mae, corr_coeff

#run_ARIMA()