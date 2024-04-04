import math

import numpy as np
from matplotlib import pyplot as plt

import matplotlib as mpl
mpl.use('TkAgg')

import Parameters
from LSTM.Create_sample_target import create_sample_gap_prediction

tete, _ = create_sample_gap_prediction(Parameters.path_test_data)

true_gap = tete.copy()[Parameters.series_prediction_start:Parameters.series_prediction_start+Parameters.length_of_prediction]

interpolated_gap = tete.copy()[Parameters.series_prediction_start-1:Parameters.series_prediction_start+Parameters.length_of_prediction+1]
interpolated_gap[1:-1] = math.nan

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

nans, x = nan_helper(interpolated_gap)
interpolated_gap[nans] = np.interp(x(nans), x(~nans), interpolated_gap[~nans])

interpolated_gap = interpolated_gap[1:-1]

### MEAN SQUARED/ABSOLUTE ERROR ###
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("Mean squared error: %.3f" % mean_squared_error(true_gap, interpolated_gap))
print("Mean absolute error: %.3f" % mean_absolute_error(true_gap, interpolated_gap))
##########################

plt.plot(true_gap)
plt.plot(interpolated_gap)
plt.show()