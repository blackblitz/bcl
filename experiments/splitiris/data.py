"""Split Iris."""

from torch.utils.data import Subset

from datasets.iris import Iris


class SplitIris:
    def train(self):
        for i in range(3):
            dataset = Iris()
            yield Subset(dataset, (dataset.y == i).nonzero()[0])

    def test(self):
        for i in range(3):
            dataset = Iris(train=False)
            yield Subset(dataset, (dataset.y == i).nonzero()[0])
