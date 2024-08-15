"""Split dataset sequences."""

from jax import random
import numpy as np
from sklearn.datasets import load_iris, load_wine
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST

from .datasets import ArrayDataset

root = 'data'
seed = 1337


def split_dataset_by_class(css, dataset):
    """Split dataset by class."""
    targets = np.array([y for x, y in dataset])
    return [Subset(dataset, np.isin(targets, cs).nonzero()[0]) for cs in css]


def split_random(key, p, dataset):
    """Split dataset randomly into two subsets."""
    whole = np.arange(len(dataset))
    selected = np.asarray(random.choice(
        key, len(dataset), shape=(int(p * len(dataset)),), replace=False
    ))
    return (
        Subset(dataset, np.setdiff1d(whole, selected)),
        Subset(dataset, selected)
    )


def split_datasets_by_class(css, train, test, validation=False):
    """Split datasets by class."""
    if validation:
        train, val = split_random(random.PRNGKey(seed), 0.2, train)
        return {
            'training': split_dataset_by_class(css, train),
            'validation': split_dataset_by_class(css, val),
            'testing': split_dataset_by_class(css, test)
        }
    return {
        'training': split_dataset_by_class(css, train),
        'testing': split_dataset_by_class(css, test)
    }


def split_iris_with_transform(transform, validation=False):
    """Split Iris dataset by class."""
    iris = load_iris()
    train = ArrayDataset(transform(iris['data']), iris['target'])
    return split_datasets_by_class(
        np.arange(3),
        *split_random(random.PRNGKey(seed), 0.2, train),
        validation=validation
    )


def split_iris(validation=False):
    """Make Split Iris."""
    return split_iris_with_transform(lambda x: x, validation=validation)


def split_iris_1(validation=False):
    """Make Split Iris 1."""
    return split_iris_with_transform(
        lambda x: x[:, 2:3], validation=validation
    )


def split_iris_2(validation=False):
    """Make Split Iris 2."""
    return split_iris_with_transform(
        lambda x: x[:, 2:4], validation=validation
    )


def split_wine(validation=False):
    """Make Split Wine."""
    wine = load_wine()
    train = ArrayDataset(wine['data'], wine['target'])
    return split_datasets_by_class(
        np.arange(3),
        *split_random(random.PRNGKey(seed), 0.2, train),
        validation=validation
    )


def split_mnist(validation=False):
    """Make Split MNIST."""

    def transform(x):
        return np.asarray(x)[:, :, None] / 255.0

    return split_datasets_by_class(
        np.arange(10).reshape((5, 2)),
        MNIST(root, download=True, transform=transform, train=True),
        MNIST(root, download=True, transform=transform, train=False),
        validation=validation
    )


def split_cifar10(validation=False):
    """Make Split CIFAR-10."""

    def transform(x):
        np.asarray(x) / 255.0

    return split_datasets_by_class(
        np.arange(10).reshape((5, 2)),
        CIFAR10(root, download=True, transform=transform, train=True),
        CIFAR10(root, download=True, transform=transform, train=False),
        validation=validation
    )
