"""Convolutional neural network for Split MNIST."""

from flax import linen as nn


class Main(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(8, (3, 3))(x)
        x = nn.swish(x)
        x = nn.max_pool(x, (8, 8), strides=(8, 8))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(8)(x)
        x = nn.swish(x)
        x = nn.Dense(10)(x)
        return x


class Consolidator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(50)(x)
        x = nn.swish(x)
        x = nn.Dense(50)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x
