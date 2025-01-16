"""Convolutional neural networks."""

from flax import linen as nn
import jax.numpy as jnp


class CNN4(nn.Module):
    """Convolutional neural network with 4 layers."""

    conv0: int
    conv1: int
    dense0: int
    dense1: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        xs = nn.Conv(self.conv0, (3, 3))(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = nn.Conv(self.conv1, (3, 3))(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense0)(xs)
        xs = nn.swish(xs)
        xs = nn.Dense(self.dense1)(xs)
        return xs
