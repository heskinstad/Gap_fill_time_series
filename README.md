<!--<h3><b>ML for imputation of time-series data on SST near Munkholmen</b></h3>-->
# TDT4900 Master Project
Python project focusing on predicting data for (generated) missing sections from the Munkholmen OceanLab buoy SST time-series.

Gap-filling through three different methods: linear interpolation, ARIMA, and RNN LSTM.
Evaluated with generated plots and the following performance metrics: mean square error, mean absolute error, correlation coefficient, and absolute correlation coefficient.

Usage: Run main.py in LSTM directory with the path set to Gap_fill_time_series. All parameters and hyperparameters are defined in the Parameters.py file.

## Examples of gap-fillings from the three methods:

### Linear interpolation
![](imgs/interpolation_example_plot.jpg?raw=true "Linear interpolation example plot")

### ARIMA
![](imgs/ARIMA_example_plot.jpg?raw=true "ARIMA example plot")

### RNN LSTM
![](imgs/LSTM_example_plot.jpg?raw=true "RNN LSTM example plot")
