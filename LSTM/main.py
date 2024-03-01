import Parameters
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
