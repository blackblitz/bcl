"""Fully-connected neural network for Split MNIST."""

from flax import linen as nn


class Main(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(100)(x)
        x = nn.swish(x)
        x = nn.Dense(100)(x)
        x = nn.swish(x)
        return nn.Dense(10)(x)


class Consolidator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(50)(x)
        x = nn.swish(x)
        x = nn.Dense(50)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x
