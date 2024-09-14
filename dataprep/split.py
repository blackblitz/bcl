"""Split dataset sequences."""

from jax import random
import numpy as np
from sklearn.datasets import load_iris, load_wine
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST

from .datasets import ArrayDataset

root = 'pytorch'
seed = 1337


def class_split(css, dataset):
    """Split dataset by class."""
    targets = np.array([y for x, y in dataset])
    return [Subset(dataset, np.isin(targets, cs).nonzero()[0]) for cs in css]


def random_split(key, p, dataset):
    """Split dataset randomly into two subsets."""
    whole = np.arange(len(dataset))
    selected = np.asarray(random.choice(
        key, len(dataset), shape=(int(p * len(dataset)),), replace=False
    ))
    return (
        Subset(dataset, np.setdiff1d(whole, selected)),
        Subset(dataset, selected)
    )


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
    css = np.arange(3)
    return (
        {
            'training': class_split(css, training_dataset),
            'validation': class_split(css, validation_dataset),
            'testing': class_split(css, testing_dataset)
        },
        {
            'classes': dataset.target_names.tolist(),
            'features': dataset.feature_names,
            'input_shape': [len(dataset.feature_names)],
            'length': len(css)
        }
    )

def split_iris():
    """Make Split Iris."""
    return task_sequence_1(load_iris)


def split_wine():
    """Make Split Wine."""
    return task_sequence_1(load_wine)


def task_sequence_2(dataset_constructor, transform, **kwargs):
    """Generate a task sequence of pattern 2."""
    css = np.arange(10).reshape((5, 2))
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
            'training': class_split(css, training_dataset),
            'validation': class_split(css, validation_dataset),
            'testing': class_split(css, testing_dataset)
        },
        {
            'classes': testing_dataset.classes,
            'input_shape': testing_dataset.data.shape[1:],
            'length': len(css)
        }
    )


def split_mnist():
    """Make Split MNIST."""
    return task_sequence_2(MNIST, lambda x: np.asarray(x)[:, :, None] / 255.0)


def split_cifar10():
    """Make Split MNIST."""
    return task_sequence_2(CIFAR10, lambda x: np.asarray(x) / 255.0)
