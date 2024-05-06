"""Split Iris."""

from collections.abc import Sequence

from torch.utils.data import Subset

from ..datasets.iris import Iris


class SplitIris(Sequence):
    def __init__(self, train=True):
        self.dataset = Iris(train=train)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index not in range(self.__len__()):
            raise IndexError()
        return Subset(self.dataset, (self.dataset.y == index).nonzero()[0])


class SplitIris1(SplitIris):
    def __init__(self, train=True):
        self.dataset = Iris(
            transform=lambda x: x[[2]],
            target_transform=lambda x: x == 2,
            train=train
        )


class SplitIris2(SplitIris):
    def __init__(self, train=True):
        self.dataset = Iris(
            transform=lambda x: x[[2, 3]],
            train=train
        )
