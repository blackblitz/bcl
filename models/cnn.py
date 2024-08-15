"""Convolutional neural networks."""

from flax import linen as nn
import jax.numpy as jnp
import toolz


class CNN4(nn.Module):
    """CNN with 4 layers."""

    conv0: int
    conv1: int
    dense0: int
    dense1: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        return toolz.pipe(
            xs,
            nn.Conv(self.conv0, (3, 3)),
            nn.relu,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(self.conv1, (3, 3)),
            nn.relu,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            toolz.curry(jnp.reshape, shape=(xs.shape[0], -1)),
            nn.Dense(self.dense0),
            nn.relu,
            nn.Dense(self.dense1)
        )


class CNN7(nn.Module):
    """CNN with 7 layers."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    dense0: int
    dense1: int
    dense2: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        return toolz.pipe(
            xs,
            nn.Conv(self.conv0, (3, 3)),
            nn.relu,
            nn.Conv(self.conv1, (3, 3)),
            nn.relu,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(self.conv2, (3, 3)),
            nn.relu,
            nn.Conv(self.conv3, (3, 3)),
            nn.relu,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            toolz.curry(jnp.reshape, shape=(xs.shape[0], -1)),
            nn.Dense(self.dense0),
            nn.relu,
            nn.Dense(self.dense1),
            nn.relu,
            nn.Dense(self.dense2)
        )
