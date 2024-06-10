<!--<h3><b>ML for imputation of time-series data on SST near Munkholmen</b></h3>-->
# TDT4900 Master Project
Python project focusing on predicting data for (generated) missing sections from the Munkholmen OceanLab buoy SST time-series.

Gap-filling through three different methods: linear interpolation, ARIMA, and RNN LSTM.
Evaluated with generated plots and the following performance metrics: mean square error, mean absolute error, correlation coefficient, and absolute correlation coefficient.

Usage: Run main.py in LSTM directory with the path set to Gap_fill_time_series. All parameters and hyperparameters are defined in the Parameters.py file.

## Examples of gap-fillings from the three methods:

### Linear interpolation
!(imgs/our_model/linear_interpolation_example.png?raw=true "Linear interpolation example plot")

### ARIMA
!(imgs/our_model/ARIMA_example.png?raw=true "ARIMA example plot")

### RNN LSTM
!(imgs/RNN_LSTM_example.png?raw=true "RNN LSTM example plot")
