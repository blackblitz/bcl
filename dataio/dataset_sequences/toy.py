"""Toy datasets."""

from jax import random
import numpy as np
from sklearn.datasets import load_iris

from ..datasets import ArrayDataset
from .split import split_random

seed = 1337


def iris_2(validation=False):
    """Make the singleton Iris 2 dataset sequence."""
    iris = load_iris()
    train = ArrayDataset(iris['data'][:, 2: 4], iris['target'])
    train, test = split_random(random.PRNGKey(seed), 0.2, train)
    return {'training': [train], 'testing': [test]}


def sinusoid(validation=False):
    """Make the sinusoid dataset sequence from https://github.com/timrudner/S-FSVI."""
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
    size = 20
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

    dataset_sequences = {
        'training': make_dataset_sequence(samples[0], samples[1]),
        'validation': make_dataset_sequence(samples[2], samples[3]),
        'testing': make_dataset_sequence(samples[4], samples[5])
    }
    return (
        dataset_sequences
        if validation
        else {k: dataset_sequences[k] for k in ['training', 'testing']}
    )
