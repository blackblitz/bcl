"""Split MNIST."""

from collections.abc import Sequence

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import MNIST


class SplitMNIST(Sequence):
    def __init__(self, train=True):
        self.dataset = MNIST(
            'data', download=True, train=train,
            transform=lambda x: np.asarray(x)[:, :, None] / 255.0
        )

    def __len__(self):
        return 5

    def __getitem__(self, index):
        if index not in range(self.__len__()):
            raise IndexError()
        return Subset(
            self.dataset,
            np.isin(
                self.dataset.targets, [2 * index, 2 * index + 1]
            ).nonzero()[0]
        )
