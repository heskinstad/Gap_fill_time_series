import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Create_sample_target import create_sample_target
from Dataset_loader import dataset_loader
from Network_model_lstm_rnn import network_model_lstm_rnn
from Trainer import Trainer


def train_model():
    samples, targets = create_sample_target("data/Train/Daily-train.csv", 60, 75)

    # Create tensors from data arrays
    tensor_samples = torch.from_numpy(samples).float()
    tensor_targets = torch.from_numpy(targets).float()

    print(tensor_samples.shape)
    print(tensor_targets.shape)

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
        optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    )

    # Train the model
    trainer.train(epochs=10)

    # Save the model
    torch.save(model.state_dict(), "Trained_models/trained_model_lstm_rnn.pt")
