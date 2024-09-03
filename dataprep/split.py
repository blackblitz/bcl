"""Split dataset sequences."""

from jax import random
import numpy as np
from sklearn.datasets import load_iris, load_wine
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST

from .datasets import ArrayDataset

root = 'pytorch'
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


def split_iris(validation=False):
    """Make Split Iris."""
    iris = load_iris()
    train = ArrayDataset(iris['data'], iris['target'])
    css = np.arange(3)
    return (
        split_datasets_by_class(
            css,
            *split_random(random.PRNGKey(seed), 0.2, train),
            validation=validation
        ),
        {
            'classes': iris.target_names.tolist(),
            'features': iris.feature_names,
            'input_shape': [len(iris.feature_names)],
            'length': len(css)
        }
    )


def split_wine(validation=False):
    """Make Split Wine."""
    wine = load_wine()
    train = ArrayDataset(wine['data'], wine['target'])
    css = np.arange(3)
    return (
        split_datasets_by_class(
            css,
            *split_random(random.PRNGKey(seed), 0.2, train),
            validation=validation
        ),
        {
            'classes': wine.target_names.tolist(),
            'features': wine.feature_names,
            'input_shape': [len(wine.feature_names)],
            'length': len(css)
        }
    )


def split_mnist(validation=False):
    """Make Split MNIST."""

    def transform(x):
        return np.asarray(x)[:, :, None] / 255.0

    css = np.arange(10).reshape((5, 2))
    training = MNIST(root, download=True, transform=transform, train=True)
    testing = MNIST(root, download=True, transform=transform, train=False)
    return (
        split_datasets_by_class(
            css, training, testing, validation=validation
        ),
        {
            'classes': training.classes,
            'input_shape': training.data.shape[1:],
            'length': len(css)
        }
    )



def split_cifar10(validation=False):
    """Make Split CIFAR-10."""

    def transform(x):
        return np.asarray(x) / 255.0

    css = np.arange(10).reshape((5, 2))
    training = CIFAR10(root, download=True, transform=transform, train=True)
    testing = CIFAR10(root, download=True, transform=transform, train=False)
    return (
        split_datasets_by_class(
            css, training, testing, validation=validation
        ),
        {
            'classes': training.classes,
            'input_shape': training.data.shape[1:],
            'length': len(css)
        }
    )
