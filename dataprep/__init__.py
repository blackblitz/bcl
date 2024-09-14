"""Dataset sequences."""

from .singleton import (
    cifar10, cifar100, emnist_letters, fashionmnist, mnist, svhn,
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
    'mnist',
    'sinusoid',
    'split_cifar10',
    'split_iris',
    'split_mnist',
    'split_wine',
    'svhn'
]
