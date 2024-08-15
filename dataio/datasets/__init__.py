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


def to_arrays(dataset, apply=lambda x: x, path='data.npy', memmap=False):
    """Convert a dataset to arrays."""
    dtype = dataset[0][0].dtype
    shape = (
        len(dataset),
        *apply(np.expand_dims(dataset[0][0], 0))[0].shape
    )
    xs = np.lib.format.open_memmap(
        path, mode='w+', dtype=dtype, shape=shape
    ) if memmap else np.empty(shape, dtype=dtype)
    ys = np.empty(len(dataset), dtype=np.int64)
    for i, (x, y) in enumerate(dataset):
        xs[i] = apply(np.expand_dims(x, 0))[0]
        ys[i] = y
    if memmap:
        xs = np.load(path, mmap_mode='r')
    return xs, ys
