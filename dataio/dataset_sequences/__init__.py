"""Dataset sequences."""

from .toy import iris_2, sinusoid
from .split import (
    split_iris, split_iris_1, split_iris_2, split_mnist, split_cifar10
)

__all__ = [
    'iris_2',
    'sinusoid',
    'split_iris',
    'split_iris_1',
    'split_iris_2',
    'split_mnist',
    'split_cifar10'
]
