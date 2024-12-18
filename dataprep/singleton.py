"""Singleton task sequences."""

from pathlib import Path

from jax import random
import numpy as np
from PIL import Image
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from torchvision.datasets import (
    CIFAR10, CIFAR100, EMNIST, FashionMNIST, MNIST, SVHN
)

from . import pytorch_src, sd198_src, seed
from .datasets import ArrayDataset
from .split import random_split


def task_sequence_load(load):
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


def task_sequence_constructor(
    constructor,
    transform=lambda x: x,
    target_transform=lambda x: x,
    **kwargs
):
    """Generate a task sequence from a PyTorch Dataset constructor."""

    training_dataset = constructor(
        pytorch_src, download=True, transform=transform,
        target_transform=target_transform, train=True, **kwargs
    )
    training_dataset, validation_dataset = random_split(
        random.PRNGKey(seed), 0.2, training_dataset
    )
    testing_dataset = constructor(
        pytorch_src, download=True, transform=transform,
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
    return task_sequence_constructor(MNIST, transform=transform_grayscale)


def emnist_letters():
    """Make MNIST."""
    return task_sequence_constructor(
        EMNIST,
        transform=transform_grayscale,
        target_transform=lambda x: x - 1,
        split='letters'
    )


def fashionmnist():
    """Make MNIST."""
    return task_sequence_constructor(FashionMNIST, transform=transform_grayscale)


def cifar10():
    """Make CIFAR-10."""
    return task_sequence_constructor(CIFAR10, transform=transform_rgb)


def cifar100():
    """Make CIFAR-100."""
    return task_sequence_constructor(CIFAR100, transform=transform_rgb)


def svhn():
    """Make SVHN."""
    training_dataset = SVHN(
        pytorch_src, download=True, transform=transform_rgb, split='train'
    )
    training_dataset, validation_dataset = random_split(
        random.PRNGKey(seed), 0.2, training_dataset
    )
    testing_dataset = SVHN(
        pytorch_src, download=True, transform=transform_rgb, split='test'
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
            'input_shape': testing_dataset[0][0].shape,
            'length': 1
        }
    )


def transform_resize(path):
    """Read image from path, resize and convert to numpy array."""
    return np.asarray(
        Image.open(path).resize((28, 28), resample=Image.Resampling.LANCZOS)
    ) / 255.0


def sd169():
    """Make SD-169."""
    overlap = [
        'Actinic_solar_Damage(Actinic_Cheilitis)',
        'Actinic_solar_Damage(Actinic_Keratosis)',
        'Actinic_solar_Damage(Cutis_Rhomboidalis_Nuchae)',
        'Actinic_solar_Damage(Pigmentation)',
        'Actinic_solar_Damage(Solar_Elastosis)',
        'Actinic_solar_Damage(Solar_Purpura)',
        'Actinic_solar_Damage(Telangiectasia)',
        'Basal_Cell_Carcinoma',
        "Becker's_Nevus",
        'Benign_Keratosis',
        'Blue_Nevus',
        'Compound_Nevus',
        'Congenital_Nevus',
        'Dermatofibroma',
        'Disseminated_Actinic_Porokeratosis',
        'Dysplastic_Nevus',
        'Epidermal_Nevus',
        'Halo_Nevus',
        'Junction_Nevus',
        'Lentigo_Maligna_Melanoma',
        'Leukocytoclastic_Vasculitis',
        'Linear_Epidermal_Nevus',
        'Malignant_Melanoma',
        'Nail_Nevus',
        'Nevus_Comedonicus',
        'Nevus_Incipiens',
        'Nevus_Sebaceous_of_Jadassohn',
        'Nevus_Spilus',
        'Poikiloderma_Atrophicans_Vasculare'
    ]
    paths = np.array([
        path for path in Path(sd198_src).glob('*/*.jpg')
        if not any(s in path.parent.name for s in overlap)
    ])
    classes = sorted(set(path.parent.name for path in paths))
    class_to_index = {c: i for i, c in enumerate(classes)}
    training_paths, testing_paths = train_test_split(
        paths,
        test_size=0.2,
        random_state=seed,
        stratify=np.array([
            class_to_index[path.parent.name] for path in paths
        ])
    )
    training_paths, validation_paths = train_test_split(
        training_paths,
        test_size=0.2,
        random_state=seed,
        stratify=np.array([
            class_to_index[path.parent.name] for path in training_paths
        ])
    )
    training_cats = np.array([
        class_to_index[path.parent.name] for path in training_paths
    ])
    validation_cats = np.array([
        class_to_index[path.parent.name] for path in validation_paths
    ])
    testing_cats = np.array([
        class_to_index[path.parent.name] for path in testing_paths
    ])
    training_dataset = ArrayDataset(
        training_paths, training_cats, transform=transform_resize
    )
    validation_dataset = ArrayDataset(
        validation_paths, validation_cats, transform=transform_resize
    )
    testing_dataset = ArrayDataset(
        testing_paths, testing_cats, transform=transform_resize
    )
    return (
        {
            'training': [training_dataset],
            'validation': [validation_dataset],
            'testing': [testing_dataset]
        }, 
        {
            'classes': classes,
            'input_shape': testing_dataset[0][0].shape,
            'length': 1
        }
    )
