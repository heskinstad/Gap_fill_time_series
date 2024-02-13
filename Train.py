import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Parameters
from Create_sample_target import create_sample_target_training
from Dataset_loader import dataset_loader
from Network_model_lstm_rnn import network_model_lstm_rnn
from Trainer import Trainer


def train_model():
    samples, targets = create_sample_target_training("data/Train/Daily-train.csv")

    # Create tensors from data arrays
    tensor_samples = torch.from_numpy(samples).float()
    tensor_targets = torch.from_numpy(targets).float()

    print("Tensor shapes:")
    print("Samples: " + str(tensor_samples.shape))
    print("Targets: " + str(tensor_targets.shape))

    # Put samples and targets into a dataset
    dataset = dataset_loader(tensor_samples, tensor_targets)

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = network_model_lstm_rnn()

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
    trainer.train(epochs=10)

    # Save the model
    torch.save(model.state_dict(), "Trained_models/trained_model_lstm_rnn.pt")
