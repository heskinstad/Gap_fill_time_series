import scipy.stats as stats, csv
from LSTM.Process_csv import process_csv_column

water_munkholmen = process_csv_column('data/Munkholmen/all_hourly_fixed.csv', 2, True)
air_lade = process_csv_column('data/Munkholmen/Munkholmen_air_temp_hourly.csv', 2, True)

print(len(water_munkholmen))
print(len(air_lade))

r = stats.pearsonr(water_munkholmen[:], air_lade[:])[0]

print('r = %.3f' % (r))