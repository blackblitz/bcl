"""Data sequences."""

import numpy as np
from jax import random
from jax.tree_util import tree_map
from torch.utils.data import default_collate


def shuffle_batch(key, size, x, y):
    indices = random.permutation(key, len(y))
    for i in range(0, len(indices), size):
        idx = indices[i : i + size]
        yield x[idx], y[idx]


def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))
