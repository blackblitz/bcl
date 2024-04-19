"""Split Iris 2."""

from torch.utils.data import Subset

from torchds.datasets.iris import Iris


class SplitIris2:
    def train(self):
        for i in range(3):
            dataset = Iris(transform=lambda x: x[[2, 3]])
            yield Subset(dataset, (dataset.y == i).nonzero()[0])

    def test(self):
        for i in range(3):
            dataset = Iris(transform=lambda x: x[[2, 3]], train=False)
            yield Subset(dataset, (dataset.y == i).nonzero()[0])
