import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Parameters
from Create_sample_target import create_sample_target_training, create_sample_target_gap_training
from Dataset_loader import dataset_loader
from Network_model_lstm_rnn import network_model_lstm_rnn
from Trainer import Trainer
from Produce_parameters_log import Produce_log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    if Parameters.prediction_mode == "forecast_forward":
        samples, targets = create_sample_target_training(Parameters.path_train_data)
    else:
        samples, targets = create_sample_target_gap_training(Parameters.path_train_data, Parameters.path_train_data_other_variable)

    # Create tensors from data arrays
    tensor_samples = torch.from_numpy(samples).float()
    tensor_targets = torch.from_numpy(targets).float()

    print("Tensor shapes:")
    print("Samples: " + str(tensor_samples.shape))
    print("Targets: " + str(tensor_targets.shape))

    # Put samples and targets into a dataset
    dataset = dataset_loader(tensor_samples, tensor_targets)

    train_dataloader = DataLoader(dataset, batch_size=Parameters.batch_size, shuffle=True)

    # Initialize model
    model = network_model_lstm_rnn().to(device)

    # Create a trainer
    trainer = Trainer(
        model=model,
        dataset=train_dataloader,
        loss_fn=nn.MSELoss(),
        optimizer=optim.SGD(
            model.parameters(),
            lr=Parameters.learning_rate,
            momentum=Parameters.momentum,
            weight_decay=Parameters.weight_decay)
    )

    # Train the model
    trainer.train(epochs=Parameters.epochs)

    # Save the model
    torch.save(model.state_dict(), Parameters.path_trained_model)

    # Produce training log file
    Produce_log()
