import Parameters, os


# Writes a log of parameters used in latest trained model
# Because it's neat to have when trying to replicate previous results ;)
def Produce_log(loss):
    model_name = Parameters.model_name

    multiple_variables = "Multiple variables: " + str(Parameters.multiple_variables)
    input_size = "Input size: " + str(Parameters.input_size)

    dataset = "Dataset: " + Parameters.path_train_data
    dataset2 = "Dataset 2nd variable: " + Parameters.path_train_data_other_variable
    dataset3 = "Dataset 3rd variable: " + Parameters.path_train_data_other_variable2
    column_or_row = "Column or row: " + Parameters.column_or_row
    index = "Column index: " + str(
        Parameters.column_index) if Parameters.column_or_row == "column" else "Row index: " + str(Parameters.row_index)
    index_2nd = "Column index 2nd variable: " + str(Parameters.column_index_second_variable)
    index_3rd = "Column index 3rd variable: " + str(Parameters.column_index_third_variable)

    prediction_mode = "Prediction mode: " + str(Parameters.prediction_mode)
    lookback = "Lookback: " + str(Parameters.lookback)
    length_of_prediction = "Length of prediction (gap size): " + str(Parameters.length_of_prediction)
    lookforward = "Lookforward: " + str(Parameters.lookforward)
    train_on_entire_series = "Train on entire series: " + str(Parameters.train_on_entire_series)
    num_of_sample_targets_per_series = "Number of sample-targets per series: " + str(
        Parameters.num_of_sample_targets_per_series)
    total_num_of_series = "Total number of series: " + str(Parameters.total_num_of_series)

    epochs = "Epochs: " + str(Parameters.epochs)
    learning_rate = "Learning rate: " + str(Parameters.learning_rate)
    momentum = "Momentum: " + str(Parameters.momentum)
    weight_decay = "Weight decay: " + str(Parameters.weight_decay)
    batch_size = "Batch size: " + str(Parameters.batch_size)

    num_layers = "Number of layers: " + str(Parameters.num_layers)
    hidden_layer_size = "Hidden layer size: " + str(Parameters.hidden_layer_size)

    loss = "Loss: " + str(loss)

    f = open(os.getcwd() + r"\Trained_models\\" + model_name + "_log.txt", "w")

    f.write(
        "Name: " + model_name +
        "\n\nTraining parameters\n" +
        "-------------------\n" +
        "\nTraining file info:\n" +
        dataset + "\n" +
        dataset2 + "\n" +
        dataset3 + "\n" +
        column_or_row + "\n" +
        index + "\n" +
        index_2nd + "\n" +
        index_3rd + "\n" +
        multiple_variables + "\n" +
        input_size + "\n" +
        "\nSamples & targets creation:\n" +
        prediction_mode + "\n" +
        lookback + "\n" +
        length_of_prediction + "\n" +
        lookforward + "\n" +
        train_on_entire_series + "\n" +
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
        loss
    )

    f.close()