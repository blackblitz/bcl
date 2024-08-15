"""Dataset sequences."""

from .toy import sinusoid
from .split import (
    split_iris, split_iris_1, split_iris_2, split_mnist, split_cifar10
)

__all__ = [
    'sinusoid',
    'split_iris',
    'split_iris_1',
    'split_iris_2',
    'split_mnist',
    'split_cifar10'
]
