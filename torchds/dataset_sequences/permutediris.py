"""Permuted Iris."""

from collections.abc import Sequence

import numpy as np
from jax import random

from ..datasets.iris import Iris


class PermutedIris(Sequence):
    def __init__(self, train=True):
        self.train = train
        self.keys = random.split(random.PRNGKey(1337), num=self.__len__() - 1)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index not in range(self.__len__()):
            raise IndexError
        if index == 0:
            return Iris(train=self.train)
        return Iris(
            transform=lambda x: np.asarray(
                random.permutation(self.keys[index - 1], x)
            ),
            train=self.train
        )
