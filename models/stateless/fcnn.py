"""Fully-connected neural networks."""

from flax import linen as nn
import jax.numpy as jnp


class FCNN1(nn.Module):
    """FCNN with 1 layer."""

    dense: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense)(xs)
        return xs


class FCNN2(nn.Module):
    """FCNN with 2 layers."""

    dense0: int
    dense1: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense0)(xs)
        xs = nn.swish(xs)
        xs = nn.Dense(self.dense1)(xs)
        return xs


class FCNN3(nn.Module):
    """FCNN with 3 layers."""

    dense0: int
    dense1: int
    dense2: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense0)(xs)
        xs = nn.swish(xs)
        xs = nn.Dense(self.dense1)(xs)
        xs = nn.swish(xs)
        xs = nn.Dense(self.dense2)(xs)
        return xs
