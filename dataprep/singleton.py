"""Singleton task sequences."""

from jax import random
import numpy as np
from torchvision.datasets import (
    CIFAR10, CIFAR100, EMNIST, FashionMNIST, MNIST, SVHN
)

from .split import random_split

root = 'pytorch'
seed = 1337


def task_sequence_1(dataset_constructor, transform, **kwargs):
    """Generate an task sequence of pattern 1."""

    training_dataset = dataset_constructor(
        root, download=True, transform=transform, train=True, **kwargs
    )
    training_dataset, validation_dataset = random_split(
        random.PRNGKey(seed), 0.2, training_dataset
    )
    testing_dataset = dataset_constructor(
        root, download=True, transform=transform, train=False, **kwargs
    )
    return (
        {
            'training': [training_dataset],
            'validation': [validation_dataset],
            'testing': [testing_dataset]
        }, 
        {
            'classes': testing_dataset.classes,
            'input_shape': testing_dataset.data.shape[1:],
            'length': 1
        }
    )


def transform_grayscale(x):
    """Transform a grayscale image."""
    return np.asarray(x)[:, :, None] / 255.0


def transform_rgb(x):
    """Transform an RGB image."""
    return np.asarray(x) / 255.0


def mnist():
    """Make MNIST."""
    return task_sequence_1(MNIST, transform_grayscale)


def emnist_letters():
    """Make MNIST."""
    return task_sequence_1(EMNIST, transform_grayscale, split='letters')


def fashionmnist():
    """Make MNIST."""
    return task_sequence_1(FashionMNIST, transform_grayscale)


def cifar10():
    """Make CIFAR-10."""
    return task_sequence_1(CIFAR10, transform_rgb)


def cifar100():
    """Make CIFAR-100."""
    return task_sequence_1(CIFAR100, transform_rgb)


def svhn():
    """Make SVHN."""
    training_dataset = SVHN(
        root, download=True, transform=transform_rgb, split='train'
    )
    training_dataset, validation_dataset = random_split(
        random.PRNGKey(seed), 0.2, training_dataset
    )
    testing_dataset = SVHN(
        root, download=True, transform=transform_rgb, split='test'
    )
    return (
        {
            'training': [training_dataset],
            'validation': [validation_dataset],
            'testing': [testing_dataset]
        }, 
        {
            'classes': [
                '0 - zero',
                '1 - one',
                '2 - two',
                '3 - three',
                '4 - four',
                '5 - five',
                '6 - six',
                '7 - seven',
                '8 - eight',
                '9 - nine'
            ],
            'input_shape': testing_dataset.data.shape[1:],
            'length': 1
        }
    )
