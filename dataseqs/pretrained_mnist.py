"""Pretrained MNIST data sequences."""

import numpy as np
import jax.numpy as jnp
from jax import jit, random, vmap
from orbax.checkpoint import PyTreeCheckpointer
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from models.pretrained_mnist.foundation import state_init


class MNISTBase:
    def __init__(self):
        params = PyTreeCheckpointer().restore(
            '/home/waiyan/thesis/ckpt/pretrained_mnist/foundation'
        )
        state = state_init.replace(params=params)
        
        def transform(x):
            return np.asarray(state.apply_fn(
                {"params": state.params},
                np.asarray(x)[None, :, :, None] / 255.0,
                method=lambda m, x: m.feature_extractor(x)
            ))[0]
        
        self.train = MNIST(
            'data', train=True, download=True, transform=transform
        )
        self.test = MNIST(
            'data', train=False, download=True, transform=transform
        )


class SplitMNIST(MNISTBase):
    def trainseq(self):
        for i in range(5):
            yield Subset(
                self.train,
                np.isin(self.train.targets, [2 * i, 2 * i + 1]).nonzero()[0]
            )

    def testseq(self):
        for i in range(5):
            yield Subset(
                self.test,
                np.isin(self.test.targets, [2 * i, 2 * i + 1]).nonzero()[0]
            )
