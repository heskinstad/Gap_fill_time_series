import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta

# Load your CSV file
df = pd.read_csv('tete.csv.csv')
df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' is in datetime format

# Prepare the times for which we want to interpolate
start_date = df['time'].min()
end_date = df['time'].max()
hourly_times = pd.date_range(start=start_date, end=end_date, freq='H')

# Fit a cubic spline with the daily data
cs = CubicSpline(df['time'], df['water'])

# Interpolate the values for each hourly timestamp
hourly_water = cs(hourly_times)

# Create the hourly DataFrame
hourly_data = pd.DataFrame({
    'id': range(len(hourly_times)),
    'time': hourly_times,
    'water': hourly_water
})

# Save the hourly data to a new CSV file
hourly_data.to_csv('hourly_data_spline_interpolated.csv', index=False)
