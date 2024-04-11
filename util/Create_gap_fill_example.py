import math

import numpy as np
from matplotlib import pyplot as plt

import Parameters
from LSTM.Create_sample_target import create_sample_prediction

import matplotlib as mpl
mpl.use('TkAgg')

data, _ = create_sample_prediction(Parameters.path_train_data)

data = data[5000:6000]

gapped_data = data.copy()
gapped_data[150:200] = math.nan
gapped_data[220:243] = math.nan
gapped_data[265:275] = math.nan
gapped_data[344:384] = math.nan
gapped_data[523:663] = math.nan
gapped_data[889:923] = math.nan

plt.figure(figsize=(10, 6))
plt.plot(gapped_data, label="True data")
plt.xlabel("Time (h)")
plt.ylabel("Temperature")
plt.legend()
plt.show()


gapped_fill = np.empty(1000)
gapped_fill[:] = math.nan

gapped_fill[150:200] = data[150:200]
gapped_fill[220:243] = data[220:243]
gapped_fill[265:275] = data[265:275]
gapped_fill[344:384] = data[344:384]
gapped_fill[523:663] = data[523:663]
gapped_fill[889:923] = data[889:923]

plt.figure(figsize=(10,6))
plt.plot(gapped_data, label="True data")
plt.plot(gapped_fill, label="Predictions")
plt.xlabel("Time (h)")
plt.ylabel("Temperature")
plt.legend()
plt.show()