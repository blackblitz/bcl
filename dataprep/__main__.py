"""Script to prepare task sequences."""

import argparse
from importlib import import_module
from pathlib import Path

import orbax.checkpoint as ocp

from dataops.io import write_toml

from . import module_map
from .datasets import write_npy


def main():
    """Run the main script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of the task sequence')
    args = parser.parse_args()

    path = Path('data/prepped') / args.name
    path.mkdir(parents=True, exist_ok=True)
    ocp.test_utils.erase_and_create_empty(path)
    task_sequence, metadata = getattr(
        import_module(module_map[args.name]),
        args.name
    )()
    for split, dataset_sequence in task_sequence.items():
        for i, dataset in enumerate(dataset_sequence):
            write_npy(
                dataset,
                path / f'{split}_{i + 1}_xs.npy',
                path / f'{split}_{i + 1}_ys.npy'
            )
    write_toml(metadata, path / 'metadata.toml')


if __name__ == '__main__':
    main()
