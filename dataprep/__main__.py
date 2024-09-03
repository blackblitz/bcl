"""Script to prepare task sequences."""

import argparse
from pathlib import Path

import orbax.checkpoint as ocp

from .toy import sinusoid  # noqa: F401
from .split import (  # noqa: F401
    split_cifar10, split_iris, split_mnist, split_wine
)

from .datasets import write_npy


def main():
    """Run the main script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name of the task sequence')
    parser.add_argument(
        '-v', '--validation', help='include a validation dataset sequence',
        action='store_true'
    )
    args = parser.parse_args()

    path = Path('data') / args.name
    path.mkdir(parents=True, exist_ok=True)
    ocp.test_utils.erase_and_create_empty(path)
    task_sequence, metadata = globals()[args.name](validation=args.validation)
    for split, dataset_sequence in task_sequence.items():
        for i, dataset in enumerate(dataset_sequence):
            write_npy(
                dataset,
                path / f'{split}_xs_{i + 1}.npy',
                path / f'{split}_ys_{i + 1}.npy'
            )
    with open(path / 'metadata.toml', 'w') as file:
        for key, value in metadata.items():
            file.write(f'{key} = {value}\n')


if __name__ == '__main__':
    main()
