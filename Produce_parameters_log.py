import Parameters


# Writes a log of parameters used in latest trained model
# Because it's neat to have when trying to replicate previous results ;)
def Produce_log():
    path = "Trained_models/trained_model_lstm_rnn_params.txt"

    dataset = "Dataset: " + Parameters.path_train_data
    column_or_row = "Column or row: " + Parameters.column_or_row
    index = "Column index: " + str(Parameters.column_index) if Parameters.column_or_row == "column" else "Row index: " + str(Parameters.row_index)

    prediction_mode = "Prediction mode: " + str(Parameters.prediction_mode)
    lookback = "Lookback: " + str(Parameters.lookback)
    lookforward = "Lookforward: " + str(Parameters.lookforward)
    num_of_sample_targets_per_series = "Number of sample-targets per series: " + str(Parameters.num_of_sample_targets_per_series)
    total_num_of_series = "Total number of series: " + str(Parameters.total_num_of_series)

    epochs = "Epochs: " + str(Parameters.epochs)
    learning_rate = "Learning rate: " + str(Parameters.learning_rate)
    momentum = "Momentum: " + str(Parameters.momentum)
    weight_decay = "Weight decay: " + str(Parameters.weight_decay)
    batch_size = "Batch size: " + str(Parameters.batch_size)

    num_layers = "Number of layers: " + str(Parameters.num_layers)
    hidden_layer_size = "Hidden layer size: " + str(Parameters.hidden_layer_size)

    length_of_prediction = "Length of prediction (gap length): " + str(Parameters.length_of_prediction)

    loss = "Loss: " + "TODO"
    accuracy = "Accuracy: " + "TODO"



    f = open(path, "w")

    f.write(
        "Training parameters\n" +
        "-------------------\n" +
        "\nTraining file info:\n" +
        dataset + "\n" +
        column_or_row + "\n" +
        index + "\n" +
        "\nSamples & targets creation:\n" +
        prediction_mode + "\n" +
        lookback + "\n" +
        lookforward + "\n" +
        num_of_sample_targets_per_series + "\n" +
        total_num_of_series + "\n" +
        "\nTraining:\n" +
        epochs + "\n" +
        learning_rate + "\n" +
        momentum + "\n" +
        weight_decay + "\n" +
        batch_size + "\n" +
        "\nNetwork info:\n" +
        num_layers + "\n" +
        hidden_layer_size + "\n" +
        "\nPrediction parameters:\n" +
        length_of_prediction + "\n" +
        "\nResults after training:\n" +
        loss + "\n" +
        accuracy
    )

    f.close()