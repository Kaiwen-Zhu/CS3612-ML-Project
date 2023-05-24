import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import torch
from data import get_data


class FashionMNISTDataset(Dataset):
    """The dataset class for Fashion MNIST dataset.

    Given tuples of numpy arrays (X, Y), where X with shape (N, 1, H, W) 
    being the images and Y with shape (N,) being the labels, each item 
    in the dataset will be a tuple of an image and its label as tensors.
    """    

    def __init__(self, dataset: Tuple[np.ndarray]):
        super().__init__()
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[0][index]).to(self.device), \
            torch.from_numpy(np.array(self.dataset[1][index])).to(self.device)
    

def get_dataloader(data_root: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:    
    """Returns the data loaders of training, validation and test sets."""    
    
    X_train, X_test, Y_train, Y_test = get_data(data_root)
    nsamples = len(X_train)
    prop = 0.8  # proportion of training set
    train_dataset = FashionMNISTDataset((X_train[:int(nsamples*prop)], Y_train[:int(nsamples*prop)]))
    val_dataset = FashionMNISTDataset((X_train[int(nsamples*prop):], Y_train[int(nsamples*prop):]))
    test_dataset = FashionMNISTDataset((X_test, Y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader
