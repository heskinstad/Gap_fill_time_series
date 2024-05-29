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
data_orig = pd.read_csv('data/Munkholmen/Munkholmen_water_air_Rate_water.csv')

data['water'] = np.sqrt(data['water'])
data['water'] = data['water'].dropna()
data['water'] = data['water'].diff().dropna()
data.loc[0, 'water'] = data_orig['water'].iloc[0]

data['MET'] = np.sqrt(data['MET'])
data['MET'] = data['MET'].dropna()
data['MET'] = data['MET'].diff().dropna()
data.loc[0, 'MET'] = data_orig['MET'].iloc[0]

#data = pd.DataFrame({'water': df_water, 'air': df_air, 'MET': df_MET, 'waterflow': df_waterflow}, include_lowest=True)
#data['water'] = df_water
#data['MET'] = df_MET

result=adfuller(data['MET'])
print('Test Statistic: %f' %result[0])
print('p-value: %f' %result[1])
print('Critical values:')
for key, value in result[4].items ():
     print('\t%s: %.3f' %(key, value))

gc_res = grangercausalitytests(x=data[['water', 'MET']], maxlag=12)