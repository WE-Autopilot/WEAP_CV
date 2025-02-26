"""
Encapsulates DataLoader-related logic, including splitting and parallel loading.
"""
import torch
from torch.utils.data import DataLoader, random_split

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """
    Creates and returns a DataLoader
    :param dataset: torch.utils.data.Dataset
    :param batch_size: batch size for training
    :param shuffle: whether to shuffle the data order
    :param num_workers: number of worker processes
    :return: DataLoader
    """
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader

def split_dataset(dataset, train_ratio=0.7, test_ratio=0.15, seed=42):
    """
    Splits a dataset into train, test, and validation subsets.
    :param dataset: torch.utils.data.Dataset
    :param train_ratio: proportion of training set
    :param test_ratio: proportion of test set
    :param seed: random seed for reproducibility
    :return: (train_dataset, test_dataset, val_dataset)
    """
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = int(dataset_size * test_ratio)
    val_size = dataset_size - train_size - test_size

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    return random_split(dataset, [train_size, test_size, val_size])

def create_train_val_test_loaders(dataset, batch_size=8, train_ratio=0.7, test_ratio=0.15, num_workers=4):
    """
    Combines dataset splitting and DataLoader creation.
    :return: (train_loader, test_loader, val_loader)
    """
    train_dataset, test_dataset, val_dataset = split_dataset(dataset, train_ratio, test_ratio)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = create_dataloader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader   = create_dataloader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, val_loader