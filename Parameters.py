import os

mode = "accuracy"  # "train" or "predict" or "accuracy"
prediction_mode = "fill_gap"  # "forecast_forward to predict future states only, "fill_gap" to use data before and after gap to predict fill
multiple_variables = True

if multiple_variables:
    input_size = 2
    multivariate_str = "multivariate_"
else:
    input_size = 1
    multivariate_str = ""

# Paths data
path_train_data = os.getcwd() + r"\data\Munkholmen\2022-2023all_hourly.csv"  # Path to training dataset
path_train_data_other_variable = os.getcwd() + r"\data\MET\MET_Munkholmen_temp_hourly_2022-2023.csv"  # Path to training dataset (variable #2)
path_test_data = os.getcwd() + r"\data\Munkholmen\2024all_hourly.csv"  # Path to test dataset
path_test_data_other_variable = os.getcwd() + r"\data\MET\MET_Munkholmen_temp_hourly_2024.csv"  # Path to test dataset (variable #2)

column_or_row = "column"  # If each entry in the dataset is formatted through columns or rows
row_index = 1  # The row index to use, if entries ordered by rows
column_index = 2  # The column index to use, if entries ordered by columns
column_index_second_variable = 2
make_backup = True

# Create sample-targets
lookback = 50  # Input dimension
lookforward = 50  # Input dimension if prediction_mode is "fill_gap"
num_of_sample_targets_per_series = 14500  # Number of samples (and corresponding targets) per complete data series
total_num_of_series = 1  # Number of data series, if the data is split between multiple rows/columns

# Normalize data
normalize_values = False
data_max_value = 30
data_min_value = -10

# Training
epochs = 2000
learning_rate = 0.0032
momentum = 0.9
weight_decay = 0.0005
batch_size = 32

# Sample for prediction
prediction_series_row = 1  # Which row from the dataset file to create samples from (if ordered by rows)
prediction_series_column = 2  # Which column from the dataset file to create samples from (if ordered by columns)
prediction_series_column_second_variable = 2
series_prediction_start = 840  # The starting point of the test dataset to predict from

# Prediction parameters
length_of_prediction = 50  # Size of gap, predict all at once (batch) - BATCH MODE IS AUTOMATICALLY CHOSEN IF THIS VARIABLE IS GREATER THAN 1
number_of_predicts = 1000  # Size of gap, predict one by one (iterative)

# Network
num_layers = 1  # Number of hidden layers in the network --> Usually 1
hidden_layer_size = 300  # Usually 300
if prediction_mode == "forecast_forward":
    network_output_size = length_of_prediction
else:
    network_output_size = length_of_prediction

# Path trained model
model_name = "trained_model_lstm_rnn_munkholmen_{0}_{1}_{2}_{3}{4}_ep".format(lookback, length_of_prediction, lookforward, multivariate_str, epochs)
path_trained_model = os.getcwd() + r"\Trained_models\\" + model_name + ".pt"  # Path to trained model

# Accuracy testing
test_type = "interpolation"  # "LSTM" or "ARIMA" or "interpolation"
number_of_tests = 1000
accuracy_tests_from_array = False
test_positions = [160, 360, 560, 760, 960, 1160, 1360]
plot_every_test = True
error_every_test = True

# ARIMA parameters
p = lookback
d = 1
q = lookback