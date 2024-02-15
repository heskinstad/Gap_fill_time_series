import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

def plot_data(original_data, predicted_data):
    plt.plot(original_data, c='b')
    plt.plot(predicted_data, c='r')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()