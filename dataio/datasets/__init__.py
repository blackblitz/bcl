"""Datasets."""

import numpy as np


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
