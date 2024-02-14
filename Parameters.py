# Paths
path_train_data = "data/Train/Daily-train.csv"
path_test_data = "data/Train/Daily-train.csv"
path_trained_model = "Trained_models/trained_model_lstm_rnn.pt"

# Create sample-targets
lookback = 20
num_of_sample_targets_per_series = 1
total_num_of_series = 1

# Training
epochs = 1000
learning_rate = 0.0001
momentum = 0.9
weight_decay = 0.0005
batch_size = 8

# Sample for prediction
prediction_series = 1
series_prediction_start = 60

# Prediction parameters
length_of_prediction = 1
number_of_predicts = 10