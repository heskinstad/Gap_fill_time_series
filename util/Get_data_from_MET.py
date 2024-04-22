#!/usr/bin/env python
# Read data from an opendap server
import netCDF4

# Data from MET, at Munkholmen position in correct data period

# specify an url, the JARKUS dataset in this case
#url = 'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be?lat[451:1:451][956:1:956],lon[451:1:451][956:1:956],temperature[45325:1:61704][1:1:1][451:1:451][956:1:956]'
#url = 'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be?temperature[42229:1:58416][1:1:1][451:1:451][956:1:956],time[42229:1:42229]'
url = 'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be?temperature[42229:1:42329][1:1:1][451:1:451][956:1:956]'
# for local windows files, note that '\t' defaults to the tab character in python, so use prefix r to indicate that it is a raw string.
#url = r'f:\opendap\rijkswaterstaat\jarkus\profiles\transect.nc'
# create a dataset object
dataset = netCDF4.Dataset(url)

# lookup a variable
temp = dataset.variables['temperature']
#variable2 = dataset.variables['lat']
#variable3 = dataset.variables['lon']
# print the first 10 values
print(temp[:])
#print(dataset.variables['time'][0])
#print(variable2[0])
#print(variable3[0])

import pandas as pd

