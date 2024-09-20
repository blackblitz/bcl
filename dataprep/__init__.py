"""Dataset sequences."""

from .singleton import (
    cifar10, cifar100, emnist_letters, fashionmnist, iris, mnist, svhn, wine
)
from .split import (
    split_cifar10, split_iris, split_mnist, split_wine
)
from .toy import sinusoid

__all__ = [
    'cifar10',
    'cifar100',
    'emnist_letters',
    'fashionmnist',
    'iris',
    'mnist',
    'sinusoid',
    'split_cifar10',
    'split_iris',
    'split_mnist',
    'split_wine',
    'svhn',
    'wine'
]
