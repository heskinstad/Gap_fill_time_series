import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

from Process_csv import process_csv_row, process_csv_column
from Create_sample_target import create_sample_target

# Import data
#data = process_csv_row("data/Train/Daily-train.csv", 17)
data, data2 = create_sample_target("data/Train/Daily-train.csv", 60, 75)
data = data[0]
plt.plot(data)
plt.xlabel("Index")
plt.ylabel("Value")

plt.show()