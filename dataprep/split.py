"""Split dataset sequences."""

from jax import random
import numpy as np
from sklearn.datasets import load_iris, load_wine
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST

from . import medmnist_src, pytorch_src, seed
from .datasets import ArrayDataset


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


def task_sequence_load(load, css):
    """Generate a task sequence from scikit-learn load function."""
    dataset = load()
    training_dataset = ArrayDataset(dataset['data'], dataset['target'])
    training_dataset, testing_dataset = random_split(
        random.key(seed), 0.2, training_dataset
    )
    training_dataset, validation_dataset = random_split(
        random.key(seed), 0.2, training_dataset
    )
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
            'input_min': np.min(
                [x for x, y in training_dataset],
                axis=0
            ).tolist(),
            'input_max': np.max(
                [x for x, y in training_dataset],
                axis=0
            ).tolist(),
            'length': len(css)
        }
    )


def split_iris():
    """Make Split Iris."""
    return task_sequence_load(load_iris, np.arange(3))


def split_iris_2():
    """Make Split Iris 2."""
    def load():
        dataset = load_iris()
        dataset['data'] = dataset['data'][:, 2:4]
        dataset.feature_names = dataset.feature_names[2:4]
        return dataset

    return task_sequence_load(load, np.arange(3))


def split_wine():
    """Make Split Wine."""
    return task_sequence_load(load_wine, np.arange(3))


def task_sequence_constructor(constructor, transform, css):
    """Generate a task sequence from a PyTorch Dataset constructor."""
    training_dataset = constructor(
        pytorch_src, download=True, transform=transform, train=True
    )
    training_dataset, validation_dataset = random_split(
        random.key(seed), 0.2, training_dataset
    )
    testing_dataset = constructor(
        pytorch_src, download=True, transform=transform, train=False
    )
    return (
        {
            'training': class_split(css, training_dataset),
            'validation': class_split(css, validation_dataset),
            'testing': class_split(css, testing_dataset)
        },
        {
            'classes': testing_dataset.classes,
            'input_shape': testing_dataset[0][0].shape,
            'length': len(css)
        }
    )


def split_mnist():
    """Make Split MNIST."""
    return task_sequence_constructor(
        MNIST,
        lambda x: np.asarray(x)[:, :, None] / 255.0,
        np.arange(10).reshape((5, 2))
    )


def split_cifar10():
    """Make Split MNIST."""
    return task_sequence_constructor(
        CIFAR10,
        lambda x: np.asarray(x) / 255.0,
        np.arange(10).reshape((5, 2))
    )


def task_sequence_medmnist(name, transform, classes, css):
    """Generate a task sequence from a MedMNIST source npz file."""
    dataset = np.load(f'{medmnist_src}/{name}.npz')
    training_dataset = ArrayDataset(
        transform(dataset['train_images']), dataset['train_labels'][:, 0]
    )
    validation_dataset = ArrayDataset(
        transform(dataset['val_images']), dataset['val_labels'][:, 0]
    )
    testing_dataset = ArrayDataset(
        transform(dataset['test_images']), dataset['test_labels'][:, 0]
    )
    return (
        {
            'training': class_split(css, training_dataset),
            'validation': class_split(css, validation_dataset),
            'testing': class_split(css, testing_dataset)
        },
        {
            'classes': classes,
            'input_shape': testing_dataset[0][0].shape,
            'length': len(css)
        }
    )


def split_dermamnist():
    return task_sequence_medmnist(
        'dermamnist',
        lambda x: np.asarray(x) / 255.0,
        [
            'actinic keratoses and intraepithelial carcinoma',
            'basal cell carcinoma',
            'benign keratosis-like lesions',
            'dermatofibroma',
            'melanoma',
            'melanocytic nevi',
            'vascular lesions'
        ],
        [[0, 1, 2], [3], [4], [5], [6]]
    )
