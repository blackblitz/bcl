"""I/O operations."""

from pathlib import Path
import tomllib

import numpy as np
import tomli_w


def clear(path):
    """Remove all files and sub-directories under a directory."""
    for p in Path(path).iterdir():
        if p.is_file():
            p.unlink()
        else:
            clear(p)
            p.rmdir()


def get_filenames(split, task_id):
    """Return the file names for a task."""
    return (f'{split}_{task_id}_xs.npy', f'{split}_{task_id}_ys.npy')


def read_task(path, split, task_id):
    """Read the memory-mapped arrays for a task."""
    return tuple(
        np.lib.format.open_memmap(path / filename, mode='r')
        for filename in get_filenames(split, task_id)
    )


def read_toml(path):
    """Read a toml file."""
    with open(path, 'rb') as file:
        return tomllib.load(file)


def write_toml(data, path):
    """Write a toml file."""
    with open(path, 'wb') as file:
        return tomli_w.dump(data, file)
