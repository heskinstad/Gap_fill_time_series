# Paths
path_train_data = "Sea_hour.csv"
path_test_data = "Sea_hour.csv"
path_trained_model = "Trained_models/trained_model_lstm_rnn.pt"
column_or_row = "column"
row_index = 1
column_index = 1

# Create sample-targets
lookback = 40  # Input dimension
num_of_sample_targets_per_series = 1500
total_num_of_series = 1

# Training
epochs = 1000
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0005
batch_size = 5

# Sample for prediction
prediction_series_row = 1
prediction_series_column = 1
series_prediction_start = 1000

# Prediction parameters
type_of_prediction = "iterative"  # Either "iterative" or "batch"
length_of_prediction = 10  # Size of gap, predict all at once (batch)
number_of_predicts = 1000  # Size of gap, iterative approach