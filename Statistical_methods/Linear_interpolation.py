import math

import numpy as np

import matplotlib as mpl
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

mpl.use('TkAgg')

import Parameters
from LSTM.Create_sample_target import create_sample_gap_prediction
from LSTM.Plot_data import plot_data

def run_linear_interpolation(start=Parameters.series_prediction_start):
    data, _, _ = create_sample_gap_prediction(Parameters.path_test_data)

    true_gap = data.copy()[start:start+Parameters.length_of_prediction]

    interpolated_gap = data.copy()[start-1:start+Parameters.length_of_prediction+1]
    interpolated_gap[1:-1] = math.nan

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = nan_helper(interpolated_gap)
    interpolated_gap[nans] = np.interp(x(nans), x(~nans), interpolated_gap[~nans])

    interpolated_gap = interpolated_gap[1:-1]

    mse = mean_squared_error(true_gap, interpolated_gap)
    mae = mean_absolute_error(true_gap, interpolated_gap)
    corr_coeff = pearsonr(true_gap, interpolated_gap)[0]

    #interpolated_gap2 = data.copy()
    #interpolated_gap2[start:start+Parameters.length_of_prediction] = interpolated_gap

    if Parameters.error_every_test:
        print("Linear Interpolation Mean squared error: %.3f" % mse)
        print("Linear Interpolation Mean absolute error: %.3f" % mae)
        print("Correlation Coefficient error: %.3f" % corr_coeff)

    if Parameters.plot_every_test:
        plot_data(data, interpolated_gap, start=start)

        #plt.plot(true_gap)
        #plt.plot(data[start-10:start+Parameters.length_of_prediction+10])
        #plt.plot(interpolated_gap)
        #empty_array = np.empty(10)
        #empty_array[:] = np.nan
        #plt.plot(np.concatenate((empty_array, interpolated_gap, empty_array)))
        #plt.show()

    return mse, mae, corr_coeff

#run_linear_interpolation()