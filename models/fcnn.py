"""Fully-connected neural networks."""

from flax import linen as nn
import jax.numpy as jnp
import toolz


class FCNN1(nn.Module):
    """FCNN with 1 layer."""
    dense: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        return toolz.pipe(
            xs,
            toolz.curry(jnp.reshape, shape=(xs.shape[0], -1)),
            nn.Dense(self.dense)
        )


class FCNN2(nn.Module):
    """FCNN with 2 layers."""
    dense0: int
    dense1: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        return toolz.pipe(
            xs,
            toolz.curry(jnp.reshape, shape=(xs.shape[0], -1)),
            nn.Dense(self.dense0),
            nn.swish,
            nn.Dense(self.dense1)
        )


class FCNN3(nn.Module):
    """FCNN with 3 layers."""
    dense0: int
    dense1: int
    dense2: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        return toolz.pipe(
            xs,
            toolz.curry(jnp.reshape, shape=(xs.shape[0], -1)),
            nn.Dense(self.dense0),
            nn.swish,
            nn.Dense(self.dense1),
            nn.swish,
            nn.Dense(self.dense2)
        )
