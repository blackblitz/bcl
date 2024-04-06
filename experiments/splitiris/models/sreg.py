"""Softmax regression for Permuted Iris."""

from flax import linen as nn


class Main(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(3)(x)


class Consolidator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(50)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x
