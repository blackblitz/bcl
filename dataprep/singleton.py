"""Singleton task sequences."""

from jax import random
import numpy as np
from sklearn.datasets import load_iris, load_wine
from torchvision.datasets import (
    CIFAR10, CIFAR100, EMNIST, FashionMNIST, MNIST, SVHN
)

from .datasets import ArrayDataset
from .split import random_split

root = 'pytorch'
seed = 1337


def task_sequence_1(load):
    """Generate a task sequence of pattern 1."""
    dataset = load()
    training_dataset = ArrayDataset(dataset['data'], dataset['target'])
    training_dataset, testing_dataset = random_split(
        random.PRNGKey(seed), 0.2, training_dataset
    )
    training_dataset, validation_dataset = random_split(
        random.PRNGKey(seed), 0.2, training_dataset
    )
    return (
        {
            'training': [training_dataset],
            'validation': [validation_dataset],
            'testing': [testing_dataset]
        }, 
        {
            'classes': dataset.target_names.tolist(),
            'features': dataset.feature_names,
            'input_shape': [len(dataset.feature_names)],
            'length': 1
        }
    )

def iris():
    """Make Iris."""
    return task_sequence_1(load_iris)


def wine():
    """Make Wine."""
    return task_sequence_1(load_wine)


def task_sequence_2(
    dataset_constructor,
    transform=lambda x: x,
    target_transform=lambda x: x,
    **kwargs
):
    """Generate an task sequence of pattern 2."""

    training_dataset = dataset_constructor(
        root, download=True, transform=transform,
        target_transform=target_transform, train=True, **kwargs
    )
    training_dataset, validation_dataset = random_split(
        random.PRNGKey(seed), 0.2, training_dataset
    )
    testing_dataset = dataset_constructor(
        root, download=True, transform=transform,
        target_transform=target_transform, train=False, **kwargs
    )
    return (
        {
            'training': [training_dataset],
            'validation': [validation_dataset],
            'testing': [testing_dataset]
        }, 
        {
            'classes': testing_dataset.classes,
            'input_shape': testing_dataset[0][0].shape,
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
    return task_sequence_2(MNIST, transform=transform_grayscale)


def emnist_letters():
    """Make MNIST."""
    return task_sequence_2(
        EMNIST,
        transform=transform_grayscale,
        target_transform=lambda x: x - 1,
        split='letters'
    )


def fashionmnist():
    """Make MNIST."""
    return task_sequence_2(FashionMNIST, transform=transform_grayscale)


def cifar10():
    """Make CIFAR-10."""
    return task_sequence_2(CIFAR10, transform=transform_rgb)


def cifar100():
    """Make CIFAR-100."""
    return task_sequence_2(CIFAR100, transform=transform_rgb)


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
