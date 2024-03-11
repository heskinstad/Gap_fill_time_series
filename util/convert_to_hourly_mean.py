import pandas as pd
import os

# Read from csv file
df = pd.read_csv(os.getcwd() + r"\..\data\Munkholmen\all.csv")

# Ensure the time column is treated as datetime type and set as index
df['_time'] = pd.to_datetime(df['_time'])
df.set_index('_time', inplace=True)

# Specifically calculate the mean for the temperature column during resampling
hourly_data = df['temperature'].resample('H').mean().reset_index()

# Display the processed data
hourly_data.to_csv(os.getcwd() + r"\..\data\Munkholmen\all2.csv")
