"""Synthetic datasets."""

from jax import random
import numpy as np

from .datasets import ArrayDataset

seed = 1337


def sinusoid():
    """Make sinusoid from https://github.com/timrudner/S-FSVI."""
    mean0 = np.array([
        [0, 0.2],
        [0.6, 0.9],
        [1.3, 0.4],
        [1.6, -0.1],
        [2.0, 0.3]
    ])
    mean1 = np.array([
        [0.45, 0],
        [0.7, 0.45],
        [1.0, 0.1],
        [1.7, -0.4],
        [2.3, 0.1],
    ])
    stddev0 = np.array([
        [0.08, 0.22],
        [0.24, 0.08],
        [0.04, 0.2],
        [0.16, 0.05],
        [0.05, 0.16]
    ])
    stddev1 = np.array([
        [0.08, 0.16],
        [0.16, 0.08],
        [0.06, 0.16],
        [0.24, 0.05],
        [0.05, 0.22]
    ])
    size = 2000
    keys = random.split(random.PRNGKey(seed), num=6)
    samples = [
        np.asarray(random.normal(key, shape=(size, 2)))
        for key in keys
    ]

    def make_dataset_sequence(sample_0, sample_1):
        return [
            ArrayDataset(
                np.concatenate(
                    [m0 + s0 * sample_0, m1 + s1 * sample_1], axis=0
                ),
                np.repeat([0, 1], size)
            ) for m0, m1, s0, s1 in zip(mean0, mean1, stddev0, stddev1)
        ]

    return (
        {
            'training': make_dataset_sequence(samples[0], samples[1]),
            'validation': make_dataset_sequence(samples[2], samples[3]),
            'testing': make_dataset_sequence(samples[4], samples[5])
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
