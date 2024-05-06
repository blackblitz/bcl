"""Data IO."""

import numpy as np
import jax
from jax import random


def iter_batches(n_epochs, batch_size, x, y, shuffle=True):
    for key in random.split(random.PRNGKey(1337), num=n_epochs):
        with jax.default_device(jax.devices('cpu')[0]):
            indices = (
                np.asarray(random.permutation(key, len(y)))
                if shuffle else np.arange(len(y))
            )
        if batch_size is None:
            yield x[indices], y[indices]
        else:
            for i in range(0, len(indices), batch_size):
                idx = indices[i : i + batch_size]
                yield x[idx], y[idx]
