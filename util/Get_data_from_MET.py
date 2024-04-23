#!/usr/bin/env python
# Read data from an opendap server
import datetime

import netCDF4
import numpy as np

# Data from MET, at Munkholmen position in correct data period

# specify an url, the JARKUS dataset in this case
#url = 'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be?lat[451:1:451][956:1:956],lon[451:1:451][956:1:956],temperature[45325:1:61704][1:1:1][451:1:451][956:1:956]'
#url = 'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be?temperature[42229:1:58416][1:1:1][451:1:451][956:1:956],time[42229:1:42229]'
url = 'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be?temperature[50000:1:58416][1:1:1][451:1:451][956:1:956],time[50000:1:58416]'
# for local windows files, note that '\t' defaults to the tab character in python, so use prefix r to indicate that it is a raw string.
#url = r'f:\opendap\rijkswaterstaat\jarkus\profiles\transect.nc'
# create a dataset object
dataset = netCDF4.Dataset(url)

# lookup a variable
temp = dataset.variables['temperature'][:]
time = dataset.variables['time'][:]

import pandas as pd

time_ts = pd.Series(np.squeeze(time))
id = np.arange(7773, 7773+len(time_ts), dtype=int)

for i in range(len(time_ts)):
    time_ts[i] = datetime.datetime.fromtimestamp(time_ts[i])

temp_ts = pd.Series(np.squeeze(temp), index=[id, time_ts])

temp_ts.to_csv("MET_temp.csv", index=True, header=False)