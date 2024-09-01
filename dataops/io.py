"""I/O operations."""

from pathlib import Path
import tomllib

import numpy as np


def zarr_to_memmap(group, xs_path, ys_path):
    """Write a zarr group to memory-mapped npy files."""
    xs = np.lib.format.open_memmap(
        xs_path, mode='w+',
        dtype=np.float32,
        shape=group['xs'].shape
    )
    ys = np.lib.format.open_memmap(
        ys_path, mode='w+',
        dtype=np.uint8,
        shape=group['ys'].shape
    )
    for i, (x, y) in enumerate(zip(group['xs'], group['ys'])):
        xs[i] = x
        ys[i] = y
    return (
        np.lib.format.open_memmap(xs_path, mode='r'),
        np.lib.format.open_memmap(ys_path, mode='r')
    )


def iter_tasks(path, split):
    """Iterate through a task sequence."""
    path = Path(path)
    length = read_metadata(path / 'metadata.toml')['length']
    for i in range(length):
        yield (
            np.lib.format.open_memmap(
                path / f'{split}_xs_{i + 1}.npy', mode='r'
            ),
            np.lib.format.open_memmap(
                path / f'{split}_ys_{i + 1}.npy', mode='r'
            )
        )


def read_metadata(path):
    """Read metadata."""
    with open(path, 'rb') as file:
        return tomllib.load(file)
