"""Iris data sequences."""

import numpy as np
import jax.numpy as jnp
from jax import random, vmap
from sklearn.datasets import load_iris
from torch.utils.data import Dataset, Subset


class Iris(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        iris = load_iris()
        mask = random.bernoulli(
            random.PRNGKey(1337), p=0.2, shape=iris['target'].shape
        )
        if train:
            mask = ~mask
        self.x = iris['data'][mask]
        self.y = iris['target'][mask]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


class PermutedIris:
    def train(self):
        key = random.PRNGKey(1337)
        for i in range(3):
            if i == 0:
                yield Iris()
            else:
                key, key1 = random.split(key)
                
                def transform(x):
                    return np.asarray(random.permutation(key1, x))
                    
                yield Iris(transform=transform)

    def test(self):
        key = random.PRNGKey(1337)
        for i in range(3):
            if i == 0:
                yield Iris(train=False)
            else:
                key, key1 = random.split(key)
                
                def transform(x):
                    return np.asarray(random.permutation(key1, x))
                    
                yield Iris(train=False, transform=transform)


class SplitIris:
    @staticmethod
    def transform(x):
        return x

    @staticmethod
    def target_transform(y):
        return y

    def train(self):
        for i in range(3):
            dataset = Iris(
                transform=self.transform,
                target_transform=self.target_transform
            )
            yield Subset(dataset, (dataset.y == i).nonzero()[0])

    def test(self):
        for i in range(3):
            dataset = Iris(
                train=False,
                transform=self.transform,
                target_transform=self.target_transform
            )
            yield Subset(dataset, (dataset.y == i).nonzero()[0])


class SplitIris1(SplitIris):
    @staticmethod
    def transform(x):
        return x[[2]]

    @staticmethod
    def target_transform(y):
        return y == 2


class SplitIris2(SplitIris):
    @staticmethod
    def transform(x):
        return x[[2, 3]]