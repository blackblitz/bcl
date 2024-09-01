"""Datasets."""

import numpy as np
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """Array dataset."""

    def __init__(self, xs, ys):
        """Initialize self."""
        self.xs = xs
        self.ys = ys

    def __len__(self):
        """Return the number of rows."""
        return len(self.ys)

    def __getitem__(self, index):
        """Get row by index."""
        return self.xs[index], self.ys[index]


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
