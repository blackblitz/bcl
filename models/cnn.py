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
            nn.swish,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(self.conv1, (3, 3)),
            nn.swish,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            toolz.curry(jnp.reshape, shape=(xs.shape[0], -1)),
            nn.Dense(self.dense0),
            nn.swish,
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
            nn.swish,
            nn.Conv(self.conv1, (3, 3)),
            nn.swish,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(self.conv2, (3, 3)),
            nn.swish,
            nn.Conv(self.conv3, (3, 3)),
            nn.swish,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            toolz.curry(jnp.reshape, shape=(xs.shape[0], -1)),
            nn.Dense(self.dense0),
            nn.swish,
            nn.Dense(self.dense1),
            nn.swish,
            nn.Dense(self.dense2)
        )


class CNN10(nn.Module):
    """CNN with 10 layers."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    conv4: int
    conv5: int
    conv6: int
    dense0: int
    dense1: int
    dense2: int

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        return toolz.pipe(
            xs,
            nn.Conv(self.conv0, (3, 3)),
            nn.swish,
            nn.Conv(self.conv1, (3, 3)),
            nn.swish,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(self.conv2, (3, 3)),
            nn.swish,
            nn.Conv(self.conv3, (3, 3)),
            nn.swish,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(self.conv4, (3, 3)),
            nn.swish,
            nn.Conv(self.conv5, (3, 3)),
            nn.swish,
            nn.Conv(self.conv6, (3, 3)),
            nn.swish,
            toolz.curry(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            toolz.curry(jnp.reshape, shape=(xs.shape[0], -1)),
            nn.Dense(self.dense0),
            nn.swish,
            nn.Dense(self.dense1),
            nn.swish,
            nn.Dense(self.dense2)
        )
