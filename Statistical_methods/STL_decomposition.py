import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

data = pd.read_csv('data/Munkholmen/Munkholmen_water_air_Rate_water.csv')

stl = STL(data['air'][400:1400], period=24)
res = stl.fit()
fig = res.plot()

fig.savefig('decomposed.png')

plt.plot(data['air'][400:1400])
plt.show()