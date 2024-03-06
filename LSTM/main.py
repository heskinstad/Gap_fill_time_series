import numpy as np
from matplotlib import pyplot as plt

import Parameters
from LSTM.Create_sample_target import create_sample_target_gap_training, create_sample_gap_prediction
from Train import train_model
from Predict import predict_iterative, predict_batch
from Plot_data import plot_data

if Parameters.mode == "train":
    train_model()
elif Parameters.mode == "predict":
    if Parameters.length_of_prediction > 1:
        original_data, prediction = predict_batch()
    else:
        original_data, prediction = predict_iterative()

    plot_data(original_data, prediction)
elif Parameters.mode == "tete":
    samples, targets = create_sample_target_gap_training(Parameters.path_train_data)

    plt.plot(samples[375])
    plt.plot(targets[375])

    plt.show()

else:
    current_series, sample = create_sample_gap_prediction(Parameters.path_test_data)
    print(sample)
    plt.plot(sample)
    plt.show()