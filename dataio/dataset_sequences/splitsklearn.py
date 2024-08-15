"""Split scikit-learn datasets."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sklearn.datasets import load_iris, load_wine
from torch.utils.data import Subset

from ..datasets.sklearn import SklearnDataset


class SplitSklearn(ABC, Sequence):
    """Split scikit-learn dataset."""

    @abstractmethod
    def __init__(self, train=True):
        """Initialize self."""
        self.dataset = None

    def __len__(self):
        """Return the length of the dataset sequence."""
        return 3

    def __getitem__(self, index):
        """Get dataset by index."""
        if index not in range(self.__len__()):
            raise IndexError()
        return Subset(self.dataset, (self.dataset.y == index).nonzero()[0])


class SplitIris(SplitSklearn):
    """Split Iris."""

    def __init__(self, train=True):
        """Initialize self."""
        self.dataset = SklearnDataset(load_iris, train=train)


class SplitIris1(SplitIris):
    """Split Iris 1."""

    def __init__(self, train=True):  # pylint: disable=super-init-not-called
        """Initialize self."""
        self.dataset = SklearnDataset(
            load_iris,
            transform=lambda x: x[[2]],
            target_transform=lambda x: x == 2,
            train=train
        )


class SplitIris2(SplitIris):
    """Split Iris 2."""

    def __init__(self, train=True):  # pylint: disable=super-init-not-called
        """Initialize self."""
        self.dataset = SklearnDataset(
            load_iris,
            transform=lambda x: x[[2, 3]],
            train=train
        )


class Iris2(Sequence):
    """Iris 2 dataset."""

    def __init__(self, train=True):
        """Initialize self."""
        self.dataset = SklearnDataset(
            load_iris,
            transform=lambda x: x[[2, 3]],
            train=train
        )

    def __len__(self):
        """Return the length of the dataset sequence."""
        return 1

    def __getitem__(self, index):
        """Get dataset by index."""
        if index not in range(self.__len__()):
            raise IndexError()
        return self.dataset


class SplitWine(SplitSklearn):
    """Split Wine."""

    def __init__(self, train=True):
        """Initialize self."""
        self.dataset = SklearnDataset(load_wine, train=train)
