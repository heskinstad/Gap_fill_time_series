import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class dataset_loader(Dataset):

    def __init__(self, samples, targets):
        #assert len(samples) == len(targets), "Samples and targets must have the same length"
        self.samples = samples
        self.targets = targets

    # Return the length of the dataset (total number of samples)
    def __len__(self):
        return len(self.samples)

    # Return a single sample from the dataset
    def __getitem__(self, idx):
        print(f"Fetching index {idx}")  # Debug print
        sample = self.samples[idx]
        target = self.targets[idx]

        return sample, target


def custom_collate_fn(batch):
    samples, targets = zip(*batch)  # Unzip the batch

    # Handling variable lengths: pad samples and targets separately
    samples_padded = pad_sequence([torch.tensor(s) for s in samples], batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence([torch.tensor(t) for t in targets], batch_first=True, padding_value=0.0)

    # No need to return lengths for variable-sized targets, unless they're used in your model.
    return samples_padded, targets_padded