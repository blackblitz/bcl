"""Split Iris 1."""

from torch.utils.data import Subset

from datasets.iris import Iris


class SplitIris1:
    def train(self):
        for i in range(3):
            dataset = Iris(
                transform=lambda x: x[[2]],
                target_transform=lambda x: x == 2
            )
            yield Subset(dataset, (dataset.y == i).nonzero()[0])

    def test(self):
        for i in range(3):
            dataset = Iris(
                transform=lambda x: x[[2]],
                target_transform=lambda x: x == 2,
                train=False
            )
            yield Subset(dataset, (dataset.y == i).nonzero()[0])
