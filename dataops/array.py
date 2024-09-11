"""Array operations."""

import numpy as np
from jax import random


def shuffle(key, length):
    """Shuffle indices."""
    return np.asarray(random.permutation(key, length))


def batch(batch_size, indices):
    """Generate index batches."""
    for i in range(0, len(indices), batch_size):
        yield indices[i: i + batch_size]
