import os

import pandas as pd
from io import StringIO

# Simulate reading from a CSV file. Replace this with the actual CSV file reading.
df = pd.read_csv(os.getcwd() + r"\..\data\Munkholmen\2024all.csv")
names = ['id', 'result', 'table', 'time', 'temperature']

# Ensure the time column is treated as datetime type and set as index
df['_time'] = pd.to_datetime(df['_time'])
df.set_index('_time', inplace=True)

# Confirm the data types after conversion
print("Data types after conversion:")
print(df.dtypes)

# Group by hour and calculate the mean temperature for each group
hourly_data = df.resample('H').mean()

# Reset index to move '_time' back to a column
hourly_data.reset_index(inplace=True)

# Display the processed data
print(hourly_data)