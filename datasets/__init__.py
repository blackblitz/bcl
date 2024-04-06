"""Datasets."""

import numpy as np
from jax import tree_util
from torch.utils.data import DataLoader, default_collate


def fetch(dataset, num_epochs, batch_size):
    if batch_size is None:
        x, y = fetch_all(dataset)
        for _ in range(num_epochs):
            yield x, y
    else:
        for _ in range(num_epochs):
            yield from fetch_iter(concat, batch_size)


def fetch_all(dataset):
    return next(iter(
        DataLoader(
            dataset, batch_size=len(dataset), collate_fn=numpy_collate
        )
    ))


def fetch_iter(dataset, batch_size):
    yield from DataLoader(
        dataset, batch_size=batch_size, collate_fn=numpy_collate
    )


def numpy_collate(batch):
    return tree_util.tree_map(np.asarray, default_collate(batch))