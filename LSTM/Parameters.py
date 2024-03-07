import os

mode = "predict"  # "train" or "predict"
prediction_mode = "fill_gap"  # "forecast_forward to predict future states only, "fill_gap" to use data before and after gap to predict fill

# Paths
path_train_data = os.getcwd() + r"\data\NDBC - Train\44040h2008 - Train.csv"  # Path to training dataset
path_test_data = os.getcwd() + r"\data\NDBC - Train\44040h2008 - Test.csv"  # Path to test dataset
path_trained_model = os.getcwd() + r"\Trained_models\trained_model_lstm_rnn_works_good_ish.pt"  # Path to trained model
column_or_row = "column"  # If each entry in the dataset is formatted through columns or rows
row_index = 1  # The row index to use, if entries ordered by rows
column_index = 13  # The column index to use, if entries ordered by columns

# Create sample-targets
lookback = 50  # Input dimension
lookforward = 50  # Input dimension if prediction_mode is "fill_gap"
num_of_sample_targets_per_series = 500  # Number of samples (and corresponding targets) per complete data series
total_num_of_series = 1  # Number of data series, if the data is split between multiple rows/columns

# Normalize data
normalize_values = False
data_max_value = 30
data_min_value = -10

# Training
epochs = 1000
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0005
batch_size = 64

# Sample for prediction
prediction_series_row = 1  # Which row from the dataset file to create samples from (if ordered by rows)
prediction_series_column = 13  # Which column from the dataset file to create samples from (if ordered by columns)
series_prediction_start = 2810  # The starting point of the test dataset to predict from

# Prediction parameters
length_of_prediction = 50  # Size of gap, predict all at once (batch) - BATCH MODE IS AUTOMATICALLY CHOSEN IF THIS VARIABLE IS GREATER THAN 1
number_of_predicts = 1000  # Size of gap, predict one by one (iterative)

# Network
num_layers = 1  # Number of hidden layers in the network
hidden_layer_size = 300
if prediction_mode == "forecast_forward":
    network_output_size = length_of_prediction
else:
    network_output_size = length_of_prediction