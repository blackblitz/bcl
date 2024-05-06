"""Linreg dataset."""

import numpy as np
from jax import random
from torch.utils.data import Dataset


class Linreg(Dataset):
    def __init__(self, key, transform=None, target_transform=None):
        self.x = np.linspace(0.0, 1.0, num=100)[:, None]
        self.y = self.x[:, 0]
        self.y += np.array(random.normal(key, shape=self.y.shape))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(x)
        return x, y
