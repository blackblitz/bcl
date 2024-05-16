"""Datasets."""

import random

import numpy as np
from jax import tree_util
import torch
from torch.utils.data import DataLoader, Dataset, default_collate


class ArrayDataset(Dataset):
    """Array dataset."""

    def __init__(self, x, y):
        """Initialize self."""
        self.x = x
        self.y = y

    def __len__(self):
        """Return the number of rows."""
        return len(self.y)

    def __getitem__(self, index):
        """Get row by index."""
        return self.x[index], self.y[index]


def load(dataset, num_epochs, batch_size):
    """Iterate batches of a dataset."""
    if batch_size is None:
        x, y = load_all(dataset)
        for _ in range(num_epochs):
            yield x, y
    else:
        for _ in range(num_epochs):
            yield from load_iter(dataset, batch_size)


def load_all(dataset):
    """Load a dataset."""
    return next(iter(
        DataLoader(
            dataset, batch_size=len(dataset), collate_fn=numpy_collate
        )
    ))


def load_iter(dataset, batch_size):
    """Iterate batches of a dataset."""
    g = torch.Generator()
    g.manual_seed(1337)
    yield from DataLoader(
        dataset, batch_size=batch_size, collate_fn=numpy_collate,
        generator=g, shuffle=True, worker_init_fn=seed_worker
    )


def numpy_collate(batch):
    """Collate as a numpy array."""
    return tree_util.tree_map(np.asarray, default_collate(batch))


def seed_worker(_):
    """Seed worker."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def memmap_dataset(
    dataset,
    apply_x=lambda x: x, apply_y=lambda y: y,
    dtype_x=np.float32, dtype_y=np.int64,
    path_x='x.npy', path_y='y.npy'
):
    """Memory-map a dataset as arrays `x` and `y`."""
    array_x = np.lib.format.open_memmap(
        path_x, mode='w+', dtype=dtype_x,
        shape=(
            len(dataset),
            *apply_x(np.expand_dims(dataset[0][0], 0))[0].shape
        )
    )
    array_y = np.lib.format.open_memmap(
        path_y, mode='w+', dtype=dtype_y,
        shape=(
            len(dataset),
            *apply_y(np.expand_dims(dataset[0][1], 0)[0]).shape
        )
    )
    for i, (x, y) in enumerate(dataset):
        array_x[i] = apply_x(np.expand_dims(x, 0))[0]
        array_y[i] = apply_y(np.expand_dims(y, 0))[0]
    return np.load(path_x, mmap_mode='r'), np.load(path_y, mmap_mode='r')
