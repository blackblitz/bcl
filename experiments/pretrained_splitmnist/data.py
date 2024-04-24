"""Pre-trained Split MNIST."""

from importlib.resources import files

import numpy as np
from orbax.checkpoint import PyTreeCheckpointer
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from .pretrain.models import make_state
from .pretrain.models.cnn import Main
from datasets.iris import Iris


class SplitMNIST:
    def __init__(self, train=True):
        params = PyTreeCheckpointer().restore(
            files('experiments.splitmnist.pretrain') / 'cnn'
        )
        state = make_state(Main()).replace(params=params)
        self.dataset = MNIST(
            'data', download=True, train=train,
            transform=lambda x: np.asarray(state.apply_fn(
                {"params": state.params}, np.asarray(x)[None, :, :, None] / 255.0,
                method=lambda m, x: m.feature_extractor(x)
            ))[0]
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
