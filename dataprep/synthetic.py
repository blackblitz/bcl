"""Synthetic datasets."""

from jax import random
import numpy as np

from .datasets import ArrayDataset

seed = 1337


def sinusoid():
    """Make sinusoid from https://github.com/timrudner/S-FSVI."""
    mean = np.array([
        [
            [0, 0.2],
            [0.6, 0.9],
            [1.3, 0.4],
            [1.6, -0.1],
            [2.0, 0.3]
        ],
        [
            [0.45, 0],
            [0.7, 0.45],
            [1.0, 0.1],
            [1.7, -0.4],
            [2.3, 0.1],
        ]
    ])
    sd = np.array([
        [
            [0.08, 0.22],
            [0.24, 0.08],
            [0.04, 0.2],
            [0.16, 0.05],
            [0.05, 0.16]
        ],
        [
            [0.08, 0.16],
            [0.16, 0.08],
            [0.06, 0.16],
            [0.24, 0.05],
            [0.05, 0.22]
        ]
    ])

    def make_dataset_sequence(key):
        size = 100
        sample = mean + sd * np.asarray(
            random.normal(key, shape=(size, *mean.shape))
        )
        return [
            ArrayDataset(
                np.concatenate([sample[:, 0, i, :], sample[:, 1, i, :]]),
                np.repeat([0, 1], size)
            ) for i in range(mean.shape[1])
        ]

    keys = random.split(random.PRNGKey(seed), num=3)
    return (
        {
            'training': make_dataset_sequence(keys[0]),
            'validation': make_dataset_sequence(keys[1]),
            'testing': make_dataset_sequence(keys[2])
        },
        {
            'classes': [0, 1],
            'input_shape': [2],
            'length': 5
        }
    )


def santong():
    """Make santong."""
    mean1 = np.array([0., 0.])
    mean2 = np.array([-1., 1.])
    mean3 = np.array([1., -1.])
    stddev = np.full((2,), 0.1)
    size = 100
    keys = random.split(random.key(1337), num=3)
    ys = np.repeat(np.arange(3), 100)

    def make_dataset_sequence(key):
        xs = np.array(random.normal(key, shape=(300, 2)))
        for i, mean in enumerate([mean1, mean2, mean3]):
            xs[i * size : (i + 1) * size] = (
                mean + stddev * xs[i * size : (i + 1) * size]
            )
        return [ArrayDataset(xs, ys)]

    return (
        {
            'training': make_dataset_sequence(keys[0]),
            'validation': make_dataset_sequence(keys[1]),
            'testing': make_dataset_sequence(keys[2])
        },
        {
            'classes': [0, 1, 2],
            'input_shape': [2],
            'length': 1
        }
    )


def triangle():
    """Make traingle."""
    mean1 = np.array([0., np.sqrt(3)])
    mean2 = np.array([-1., 0.])
    mean3 = np.array([1., 0.])
    stddev = np.full((2,), 0.1)
    size = 100
    keys = random.split(random.key(1337), num=3)
    ys = np.repeat(np.arange(3), 100)

    def make_dataset_sequence(key):
        xs = np.array(random.normal(key, shape=(300, 2)))
        for i, mean in enumerate([mean1, mean2, mean3]):
            xs[i * size : (i + 1) * size] = (
                mean + stddev * xs[i * size : (i + 1) * size]
            )
        return [ArrayDataset(xs, ys)]

    return (
        {
            'training': make_dataset_sequence(keys[0]),
            'validation': make_dataset_sequence(keys[1]),
            'testing': make_dataset_sequence(keys[2])
        },
        {
            'classes': [0, 1, 2],
            'input_shape': [2],
            'length': 1
        }
    )
