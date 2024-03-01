from torch.utils.data import Dataset

class dataset_loader(Dataset):

    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets

    # Return the length of the dataset (total number of samples)
    def __len__(self):
        return len(self.samples)

    # Return a single sample from the dataset
    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]

        return sample, target