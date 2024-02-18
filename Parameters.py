mode = "predict"  # "train" or "predict"

# Paths
path_train_data = "data/NDBC - Train/44040h2008.csv"
path_test_data = "data/NDBC - Train/44040h2008.csv"
path_trained_model = "Trained_models/trained_model_lstm_rnn.pt"
column_or_row = "column"
row_index = 1
column_index = 15

# Create sample-targets
lookback = 100  # Input dimension
num_of_sample_targets_per_series = 500
total_num_of_series = 1

# Training
epochs = 3000
learning_rate = 0.05
momentum = 0.9
weight_decay = 0.0005
batch_size = 1000

# Network
num_layers = 1

# Sample for prediction
prediction_series_row = 1
prediction_series_column = 15
series_prediction_start = 1000

# Prediction parameters
length_of_prediction = 100  # Size of gap, predict all at once (batch) - BATCH MODE IS AUTOMATICALLY CHOSEN IF THIS VARIABLE IS GREATER THAN 1
number_of_predicts = 1000  # Size of gap, predict one by one (iterative)