import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from LSTM.Process_csv import process_csv_column

#water_munkholmen = process_csv_column('data/Munkholmen/all_hourly_fixed.csv', 2, True)
#air_munkholmen = process_csv_column('data/Munkholmen/Munkholmen_air_temp_hourly.csv', 2, True)

data = pd.read_csv('data/Munkholmen/Munkholmen_water_air_Rate_water.csv')

gc_res = grangercausalitytests(x=data[['water_munkholmen', 'water_rate']], maxlag=4)

for lag in range(1, 5):  # Iterating over each lag
    ftest = gc_res[lag][0]['ssr_chi2test']  # Accessing the F-test
    print(f'At lag {lag}: Chi-squared = {ftest[0]}, p-value = {ftest[1]}')