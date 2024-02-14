from Train import train_model
from Predict import predict, predict_multiple
from Plot_data import plot_data

#train_model()

original_data, prediction = predict_multiple()

plot_data(original_data, prediction)
