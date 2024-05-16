"""Module for split datasets with 10 classes into 5 tasks."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST


class Split10(ABC, Sequence):
    """Class for split datasets with 10 classes into 5 tasks."""

    @abstractmethod
    def __init__(self, train=True):
        """Initialize self."""
        self.dataset = None

    def __len__(self):
        """Return the length of the dataset sequence."""
        return 5

    def __getitem__(self, index):
        """Get dataset by index."""
        if index not in range(self.__len__()):
            raise IndexError()
        return Subset(
            self.dataset,
            np.isin(
                self.dataset.targets, [2 * index, 2 * index + 1]
            ).nonzero()[0]
        )


class SplitMNIST(Split10):
    """Split MNIST."""

    def __init__(self, train=True):
        """Initialize self."""
        self.dataset = MNIST(
            'data', download=True, train=train,
            transform=lambda x: np.asarray(x)[:, :, None] / 255.0
        )


class SplitCIFAR10(Split10):
    """Split CIFAR-10."""

    def __init__(self, train=True):
        """Initialize self."""
        self.dataset = CIFAR10(
            'data', download=True, train=train,
            transform=lambda x: np.asarray(x) / 255.0
        )
