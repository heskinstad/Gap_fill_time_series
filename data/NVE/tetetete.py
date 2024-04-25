import pandas as pd
from datetime import datetime, timedelta

# Load your CSV file
df = pd.read_csv('tete.csv.csv')

# Prepare a list to store DataFrame chunks for each day
dataframes = []

# Generate hourly data
for index, row in df.iterrows():
    date = datetime.strptime(row['time'], '%Y-%m-%d')
    daily_water = row['water']

    # Create a small DataFrame for each day's hourly data
    hourly_list = [{'id': row['id'] * 24 + hour,
                    'time': (date + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S'),
                    'water': daily_water} for hour in range(24)]

    hourly_df = pd.DataFrame(hourly_list)
    dataframes.append(hourly_df)

# Concatenate all the small DataFrames into one
hourly_data = pd.concat(dataframes, ignore_index=True)

# Save the hourly data to a new CSV file
hourly_data.to_csv('hourly_data.csv', index=False)
