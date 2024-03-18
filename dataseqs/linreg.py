"""LinReg data sequence."""

import numpy as np
import jax.numpy as jnp
from jax import random
from torch.utils.data import Dataset


class LinReg(Dataset):
    def __init__(self, key, transform=None, target_transform=None):
        self.x = np.linspace(0.0, 1.0, num=100)[:, None]
        self.y = self.x + random.normal(key, shape=self.x.shape)
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
        

class DupLinReg:
    def train(self):
        for key in random.split(random.PRNGKey(1337), num=3):
            yield LinReg(key)

    def test(self):
        yield from self.train()