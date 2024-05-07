from datetime import datetime

import pandas as pd

from Process_csv import process_csv_column

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('TkAgg')


df = process_csv_column("../data/Munkholmen/all_hourly_fixed.csv", 2)[:100]
date = process_csv_column("../data/Munkholmen/all_hourly_fixed.csv", 1, datetimes=True)[:100]

date = pd.to_datetime(date)

plt.plot(date, df)

plt.gca().xaxis.set_major_formatter(mpl.dates.mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mpl.dates.mdates.HourLocator())
plt.gcf().autofmt_xdate()


plt.show()