"""Array operations."""

import math

import numpy as np
from jax import random


def shuffle(key, length):
    """Shuffle indices."""
    return np.asarray(random.permutation(key, length))


def batch(batch_size, indices):
    """Generate index batches."""
    for i in range(0, len(indices), batch_size):
        yield indices[i: i + batch_size]


def get_pass_size(input_shape):
    """Calculate the batch size for passing through a dataset."""
    return 2 ** math.floor(
        20 * math.log2(2) - 1 - sum(map(math.log2, input_shape))
    )


def get_n_batches(total_size, batch_size):
    """Calculate the number of batches."""
    return -(total_size // -batch_size)
