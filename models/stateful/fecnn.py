"""Feature-extracting convolutional neural networks."""

from flax import linen as nn
import jax.numpy as jnp


class FECNN4(nn.Module):
    """Feature-extracting convolutional neural network with 4 layers."""

    conv0: int
    conv1: int
    dense0: int
    dense1: int
    dropout: float

    def setup(self):
        """Set up model."""
        self.tail = FE3(
            conv0=self.conv0,
            conv1=self.conv1,
            dense=self.dense0,
            dropout=self.dropout
        )
        self.head = nn.Dense(self.dense1)

    def __call__(self, xs, train: bool):
        """Apply model."""
        xs = self.tail(xs, train=train)
        xs = self.head(xs)
        return xs


class FECNN7(nn.Module):
    """Feature-extracting convolutional neural network with 7 layers."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    dense0: int
    dense1: int
    dense2: int
    dropout: float

    def setup(self):
        """Set up model."""
        self.tail = FE6(
            conv0=self.conv0, conv1=self.conv1,
            conv2=self.conv2, conv3=self.conv3,
            dense0=self.dense0, dense1=self.dense1,
            dropout=self.dropout
        )
        self.head = nn.Dense(self.dense2)

    def __call__(self, xs, train: bool):
        """Apply model."""
        xs = self.tail(xs, train=train)
        xs = self.head(xs)
        return xs


class FE3(nn.Module):
    """A feature extractor with 2 convolution layers and 1 dense layer."""

    conv0: int
    conv1: int
    dense: int
    dropout: float

    @nn.compact
    def __call__(self, xs, train: bool):
        """Apply model."""
        xs = nn.Conv(self.conv0, (3, 3))(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = nn.Conv(self.conv1, (3, 3))(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense)(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        return xs


class FE6(nn.Module):
    """A feature extractor with 4 convolution layers and 2 dense layers."""

    conv0: int
    conv1: int
    conv2: int
    conv3: int
    dense0: int
    dense1: int
    dropout: float

    @nn.compact
    def __call__(self, xs, train: bool):
        """Apply model."""
        xs = nn.Conv(self.conv0, (3, 3))(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        xs = nn.Conv(self.conv1, (3, 3))(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = nn.Conv(self.conv2, (3, 3))(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        xs = nn.Conv(self.conv3, (3, 3))(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        xs = nn.avg_pool(xs, window_shape=(2, 2), strides=(2, 2))
        xs = jnp.reshape(xs, shape=(xs.shape[0], -1))
        xs = nn.Dense(self.dense0)(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        xs = nn.Dense(self.dense1)(xs)
        xs = nn.BatchNorm(use_running_average=not train)(xs)
        xs = nn.swish(xs)
        return xs
