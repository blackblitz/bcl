"""Datasets."""

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class ArrayDataset(Dataset):
    """Array dataset."""

    def __init__(self, xs, ys, fx=lambda x: x, fy=lambda y: y):
        """Initialize self."""
        self.xs = xs
        self.ys = ys
        self.fx = fx
        self.fy = fy

    def __len__(self):
        """Return the number of rows."""
        return len(self.ys)

    def __getitem__(self, index):
        """Get row by index."""
        return self.fx(self.xs[index]), self.fy(self.ys[index])


def write_npy(dataset, xs_path, ys_path):
    """Write a pytorch dataset to npy files."""
    xs = np.lib.format.open_memmap(
        xs_path, mode='w+',
        dtype=np.float32,
        shape=(len(dataset), *dataset[0][0].shape)
    )
    ys = np.lib.format.open_memmap(
        ys_path, mode='w+',
        dtype=np.uint8,
        shape=(len(dataset),)
    )
    for i, (x, y) in enumerate(dataset):
        xs[i] = x
        ys[i] = y


def targets(dataset):
    """Get targets of a dataset."""
    return np.array([y for _, y in dataset])


def csplit(classess, dataset):
    """Split dataset into multiple subsets by class."""
    return [
        Subset(dataset, np.isin(targets(dataset), classes).nonzero()[0])
        for classes in classess
    ]


def rsplit(seed, test_size, dataset):
    """Split dataset randomly into two subsets."""
    itrain, itest = train_test_split(
        np.arange(len(dataset)), test_size=0.2,
        random_state=seed, stratify=targets(dataset)
    )
    return (Subset(dataset, itrain), Subset(dataset, itest))


def setdsattr(dataset, name, value):
    """Set attribute of a dataset."""
    root = dataset
    while isinstance(root, Subset):
        root = root.dataset
    setattr(root, name, value)
