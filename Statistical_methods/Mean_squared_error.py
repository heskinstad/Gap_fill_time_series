from sklearn.metrics import mean_squared_error

from LSTM.Process_csv import process_csv_column

water_munkholmen = process_csv_column('data/Munkholmen/all_hourly_fixed.csv', 2, True)
air_lade = process_csv_column('data/NVE/Lade-Lufttemperatur-time.csv', 3, True)

mse = mean_squared_error(water_munkholmen[:8760], air_lade[:8760])

print(mse)


def get_mean_squared_error(array1, array2):

    return mean_squared_error(array1, array2)