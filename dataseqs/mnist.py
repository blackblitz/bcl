"""MNIST data sequences."""

import numpy as np
import jax.numpy as jnp
from jax import random, vmap
from torch.utils.data import Subset
from torchvision.datasets import MNIST


class PermutedMNIST:
    @staticmethod
    def make_transform(key):
        def nop(x):
            return np.asarray(x)[:, :, None] / 255.0
            
        def permute(x):
            return np.asarray(random.permutation(
                key, np.asarray(x), independent=True
            ))[:, :, None] / 255.0

        return nop if key is None else permute
        
    def train(self):
        key = random.PRNGKey(1337)
        for i in range(5):
            if i == 0:
                key1 = None
            else:
                key, key1 = random.split(key)
            yield MNIST(
                'data', train=True, download=True,
                transform=self.make_transform(key1)
            )

    def test(self):
        key = random.PRNGKey(1337)
        for i in range(5):
            if i == 0:
                key1 = None
            else:
                key, key1 = random.split(key)
            yield MNIST(
                'data', train=False, download=True,
                transform=self.make_transform(key1)
            )


class SplitMNIST:
    @staticmethod
    def transform(x):
        return np.asarray(x)[:, :, None] / 255.0
    
    def train(self):
        for i in range(5):
            dataset = MNIST(
                'data', train=True, download=True, transform=self.transform
            )
            yield Subset(
                dataset,
                np.isin(dataset.targets, [2 * i, 2 * i + 1]).nonzero()[0]
            )

    def test(self):
        for i in range(5):
            dataset = MNIST(
                'data', train=True, download=True, transform=self.transform
            )
            yield Subset(
                dataset,
                np.isin(dataset.targets, [2 * i, 2 * i + 1]).nonzero()[0]
            )