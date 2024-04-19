"""Torch datasets."""

import random

import numpy as np
from jax import tree_util
import torch
from torch.utils.data import DataLoader, default_collate


def fetch(dataset, num_epochs, batch_size):
    if batch_size is None:
        x, y = fetch_all(dataset)
        for _ in range(num_epochs):
            yield x, y
    else:
        for _ in range(num_epochs):
            yield from fetch_iter(dataset, batch_size)


def fetch_all(dataset):
    return next(iter(
        DataLoader(
            dataset, batch_size=len(dataset), collate_fn=numpy_collate
        )
    ))


def fetch_iter(dataset, batch_size):
    g = torch.Generator()
    g.manual_seed(1337)
    yield from DataLoader(
        dataset, batch_size=batch_size, collate_fn=numpy_collate,
        generator=g, shuffle=True, worker_init_fn=seed_worker
    )


def numpy_collate(batch):
    return tree_util.tree_map(np.asarray, default_collate(batch))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
