"""Logistic regression for Split Iris 1."""

from flax import linen as nn


class Main(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)


class Consolidator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(200)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x
