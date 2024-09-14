"""I/O operations."""

import math
from pathlib import Path
import tomllib

import numpy as np


def clear(path):
    """Remove all files and sub-directories under a directory."""
    for p in Path(path).iterdir():
        if p.is_file():
            p.unlink()
        else:
            clear(p)
            p.rmdir()


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
    length = read_toml(path / 'metadata.toml')['length']
    for i in range(length):
        yield (
            np.lib.format.open_memmap(
                path / f'{split}_{i + 1}_xs.npy', mode='r'
            ),
            np.lib.format.open_memmap(
                path / f'{split}_{i + 1}_ys.npy', mode='r'
            )
        )


def read_toml(path):
    """Read metadata."""
    with open(path, 'rb') as file:
        return tomllib.load(file)


def get_pass_size(input_shape):
    """Calculate the batch size for passing through a dataset."""
    return 2 ** math.floor(
        20 * math.log2(2) - 2 - sum(map(math.log2, input_shape))
    )
