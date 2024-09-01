"""Array operations."""

import numpy as np
import jax
from jax import random


def pass_batches(batch_size, xs, ys):
    """Yield batches of data."""
    for i in range(0, len(ys), batch_size):
        idx = slice(i, i + batch_size)
        yield xs[idx], ys[idx]


def draw_batches(key, batch_size, xs, ys):
    """Shuffle and yield batches of data."""
    with jax.default_device(jax.devices('cpu')[0]):
        indices = np.asarray(random.permutation(key, len(ys)))
    yield from pass_batches(batch_size, xs[indices], ys[indices])
