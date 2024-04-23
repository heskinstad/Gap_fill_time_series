# Augmented Dickey Fuller Test
# From: https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/
import os

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

import Parameters
from LSTM import Create_sample_target
from LSTM.Create_sample_target import create_sample_target_ARIMA

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

df, _ = Create_sample_target.create_sample_gap_prediction(os.getcwd() + r"\data\Munkholmen\all_hourly_fixed.csv")

sample_before, target, sample_after = create_sample_target_ARIMA(Parameters.path_test_data)

result = adfuller(df, autolag='AIC')
#result = adfuller(sample_before, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


plot_acf(df[:50])
plt.show()