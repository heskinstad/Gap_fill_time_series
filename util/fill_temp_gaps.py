import os

import pandas as pd

# Read the data from the string (use pd.read_csv('filename.csv') if you have a file)
df = pd.read_csv(os.getcwd() + r"\..\data\Munkholmen\all_hourly_fixed.csv")

# Convert the 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the timestamp column as the index
df.set_index('timestamp', inplace=True)

# Resample the DataFrame to ensure all hours are represented, and interpolate missing data
df_resampled = df.resample('H').mean()
df_resampled['temp'] = df_resampled['temp'].interpolate()

# Reset index to add 'timestamp' back as a column and reformat DataFrame
df_resampled.reset_index(inplace=True)
df_resampled['id'] = range(len(df_resampled))

# Print the resulting DataFrame
print(df_resampled)

# Optionally, save the DataFrame back to a CSV
df_resampled.to_csv('updated_data.csv', index=False)
