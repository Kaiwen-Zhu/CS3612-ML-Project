import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import torch
import matplotlib.pyplot as plt
from data import get_data


class LFWDataset(Dataset):
    """The dataset class for Labeled Faces in the Wild dataset."""    

    def __init__(self, dataset: np.ndarray):
        super().__init__()
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[index]).to(self.device)
    

def get_dataloader(data_root: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:    
    """Returns the data loaders of training and validation sets."""   

    X_train = get_data(data_root)
    nsamples = len(X_train)
    prop = 0.8  # proportion of training set
    train_dataset = LFWDataset(X_train[:int(nsamples*prop)])
    val_dataset = LFWDataset(X_train[int(nsamples*prop):])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
