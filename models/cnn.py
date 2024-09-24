"""Convolutional neural networks."""

from flax import linen as nn
import jax.numpy as jnp

from .fcnn import FCNN1


class CNN4(nn.Module):
    """Convolutional neural network with 4 layers."""

    conv0: int
    conv1: int
    dense0: int
    dense1: int
    activation: str

    def setup(self):
        """Set up model."""
        self.tail = FeatureExtractor3(
            conv0=self.conv0, conv1=self.conv1,
            dense=self.dense0, activation=self.activation
        )
        self.head = FCNN1(dense=self.dense1)

    def __call__(self, xs):
        """Apply model."""
        xs = self.tail(xs)
        xs = self.head(xs)
        return xs


class CNN7(nn.Module):
    """Convolutional neural network with 7 layers."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    dense0: int
    dense1: int
    dense2: int
    activation: str

    def setup(self):
        """Set up model."""
        self.tail = FeatureExtractor6(
            conv0=self.conv0, conv1=self.conv1,
            conv2=self.conv2, conv3=self.conv3,
            dense0=self.dense0, dense1=self.dense1,
            activation=self.activation
        )
        self.head = FCNN1(dense=self.dense2)

    def __call__(self, xs):
        """Apply model."""
        xs = self.tail(xs)
        xs = self.head(xs)
        return xs


class FeatureExtractor3(nn.Module):
    """A CNN with 2 convolution layers and 1 dense layer."""

    conv0: int
    conv1: int
    dense: int
    activation: str

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
        xs = nn.Dense(self.dense)(xs)
        xs = getattr(nn, self.activation)(xs)
        return xs


class FeatureExtractor6(nn.Module):
    """A CNN with 4 convolution layers and 2 dense layers."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    dense0: int
    dense1: int
    activation: str

    @nn.compact
    def __call__(self, xs):
        """Apply model."""
        xs = nn.Conv(self.conv0, (3, 3))(xs)
        xs = nn.swish(xs)
        xs = nn.Conv(self.conv1, (3, 3))(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = nn.Conv(self.conv2, (3, 3))(xs)
        xs = nn.swish(xs)
        xs = nn.Conv(self.conv3, (3, 3))(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense0)(xs)
        xs = nn.swish(xs)
        xs = nn.Dense(self.dense1)(xs)
        xs = getattr(nn, self.activation)(xs)
        return xs
