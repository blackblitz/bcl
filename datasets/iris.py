"""Iris dataset."""

from jax import random
from sklearn.datasets import load_iris
from torch.utils.data import Dataset


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
