# Augmented Dickey Fuller Test
# From: https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import grangercausalitytests

from LSTM import Create_sample_target
import matplotlib as mpl
mpl.use('TkAgg')

#df, _, _, _ = Create_sample_target.create_sample_gap_prediction(os.getcwd() + r"\data\Munkholmen\all_hourly_fixed.csv")

data = pd.read_csv('data/Munkholmen/Munkholmen_water_air_Rate_water.csv')

result=adfuller(data['water'])
print('Test Statistic: %f' %result[0])
print('p-value: %f' %result[1])
print('Critical values:')
for key, value in result[4].items ():
     print('\t%s: %.3f' %(key, value))